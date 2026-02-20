import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 폰트 강제 적용을 위해 추가
from wordcloud import WordCloud
from transformers import pipeline
import os
import time
import datetime
import re
import networkx as nx
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------
# 1. 기본 설정 및 폰트 세팅 (클라우드 환경 폰트 에러 완벽 해결)
# ---------------------------------------------------------
st.set_page_config(page_title="AI 탐사보도 시스템 (최종 고도화)", layout="wide")

# 폴더에 올린 malgun.ttf를 그래프 폰트로 강제 주입합니다.
if os.path.exists('malgun.ttf'):
    fm.fontManager.addfont('malgun.ttf')
    plt.rcParams['font.family'] = fm.FontProperties(fname='malgun.ttf').get_name()
else:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
plt.rcParams['axes.unicode_minus'] = False

st.markdown("""
    <style>
    .stApp { background-color: #F5F7F9; }
    h1 { color: #2C3E50; font-family: 'Malgun Gothic', sans-serif; font-weight: 800; border-bottom: 2px solid #3498DB; padding-bottom: 10px; }
    div[data-testid="stMetric"] { background-color: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 데이터 & 모델 로드
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("news_result_final.xlsx")
        if '일자' in df.columns:
            df['일자'] = pd.to_datetime(df['일자'].astype(str).str[:8], errors='coerce')
            df = df.dropna(subset=['일자'])
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

df = load_data()
classifier = load_model()

# ---------------------------------------------------------
# 3. 사이드바 (친절한 설명 및 UI 개선)
# ---------------------------------------------------------
st.sidebar.title("🎛️ AI 분석 & 전처리 옵션")

st.sidebar.markdown("### 🔍 특정 키워드 필터링")
st.sidebar.caption(
    "💡 **왜 쓰나요?**<br>"
    "전체 기사 중 특정 주제(예: '일자리', '범죄', '교육')만 좁혀서 분석하고 싶을 때 사용합니다.<br>"
    "단어를 입력하면 해당 단어가 포함된 기사만 추려내어 여론 트렌드와 긍/부정 비율을 다시 계산합니다.", 
    unsafe_allow_html=True
)
target_keyword = st.sidebar.text_input("분석할 단어를 입력하세요:", placeholder="예: 교육, 일자리, 범죄")

st.sidebar.markdown("---")
st.sidebar.subheader("🛑 자연어 전처리 (불용어 제거)")
st.sidebar.caption("워드클라우드 등에서 의미 없는 단어(노이즈)를 걸러냅니다.")
default_stopwords = "기자, 뉴스, 오늘, 발달장애인, 발달장애, 장애인, 생각, 사람, 사회, 최근, 시간, 지원, 센터, 대한, 위해"
user_stopwords = st.sidebar.text_area("불용어 목록 (쉼표로 구분)", value=default_stopwords)
stopword_list = [word.strip() for word in user_stopwords.split(',')]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ 네트워크 & 워드클라우드 설정")
max_words = st.sidebar.slider("분석할 핵심 단어 수", 5, 20, 10)
min_word_length = st.sidebar.slider("최소 단어 길이", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 분석 기간 설정")
if not df.empty and '일자' in df.columns:
    min_date = df['일자'].min().date()
    max_date = df['일자'].max().date()
    
    st.sidebar.info(f"📌 **선택 가능 기간 (상/하한선)**\n\n최저: {min_date}\n\n최고: {max_date}")
    
    col1, col2 = st.sidebar.columns(2)
    with col1: start_date = st.date_input("시작일", value=min_date, min_value=min_date, max_value=max_date)
    with col2: end_date = st.date_input("종료일", value=max_date, min_value=min_date, max_value=max_date)
else:
    start_date, end_date = None, None

# ---------------------------------------------------------
# 4. 메인 대시보드
# ---------------------------------------------------------
st.title("📰 AI 기반 발달장애인 뉴스 심층 분석기")

with st.spinner('⏳ 데이터를 정교하게 재분석 중입니다...'):
    time.sleep(0.3)

    if not df.empty and '일자' in df.columns:
        mask = (df['일자'].dt.date >= start_date) & (df['일자'].dt.date <= end_date)
        global_df_filtered = df.loc[mask].copy()
        if target_keyword:
            global_df_filtered = global_df_filtered[global_df_filtered['제목'].str.contains(target_keyword, na=False)]
    else:
        global_df_filtered = pd.DataFrame()

    total = len(global_df_filtered)
    pos = len(global_df_filtered[global_df_filtered['감성'] == '긍정']) if total > 0 else 0
    neg = len(global_df_filtered[global_df_filtered['감성'] == '부정']) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("조건 부합 기사", f"{total:,} 건")
    col2.metric("긍정 여론", f"{pos:,} 건", f"{round(pos/total*100, 1) if total>0 else 0}%")
    col3.metric("부정 여론", f"{neg:,} 건", f"-{round(neg/total*100, 1) if total>0 else 0}%")
    col4.metric("분석 타겟", target_keyword if target_keyword else "전체 제목")

    st.markdown("---")

    processed_docs = []
    
    if total > 0:
        for title in global_df_filtered['제목'].dropna().astype(str):
            clean_text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', title)
            tokens = clean_text.split()
            final_tokens = [word for word in tokens if len(word) >= min_word_length and word not in stopword_list]
            if final_tokens:
                processed_docs.append(" ".join(final_tokens))

    st.subheader("🕸️ 핵심 키워드 연관성 및 심층 분석")
    st.caption("선(Edge)이 굵을수록 언론에서 두 단어를 기사 제목에 함께(동시에) 많이 사용했다는 의미입니다.")
    tab_wordcloud, tab_network, tab_heatmap = st.tabs(["☁️ 워드클라우드", "🌐 연관성 네트워크", "🟪 유사도 히트맵"])

    if len(processed_docs) > 5:
        cv = CountVectorizer(max_features=max_words)
        dtm = cv.fit_transform(processed_docs)
        df_dtm = pd.DataFrame(dtm.toarray(), columns=cv.get_feature_names_out())
        corr_matrix = df_dtm.corr().fillna(0)

        with tab_wordcloud:
            text_for_wc = " ".join(processed_docs)
            font_path_wc = 'malgun.ttf' if os.path.exists('malgun.ttf') else None 
            wc = WordCloud(width=800, height=350, background_color='white', font_path=font_path_wc, colormap='viridis').generate(text_for_wc)
            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)

        with tab_network:
            G = nx.Graph()
            words = corr_matrix.columns
            for word in words: G.add_node(word)
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    weight = corr_matrix.iloc[i, j]
                    if weight > 0.1: 
                        G.add_edge(words[i], words[j], weight=weight)
            
            fig_net, ax = plt.subplots(figsize=(10, 6))
            pos_net = nx.spring_layout(G, k=0.5, seed=42)
            
            # [디자인 수정] 동그라미는 연한 보라색, 글자는 진한 검은색으로 변경하여 가독성 극대화
            nx.draw_networkx_nodes(G, pos_net, node_size=2500, node_color='#E8EAF6', edgecolors='#7B68EE', linewidths=2, ax=ax)
            nx.draw_networkx_edges(G, pos_net, width=[G[u][v]['weight']*5 for u,v in G.edges()], edge_color='#BDBDBD', ax=ax)
            # 글자가 잘 보이도록 font_color를 'black'으로 강제 지정
            nx.draw_networkx_labels(G, pos_net, font_size=13, font_color='black', font_weight='bold', ax=ax)
            
            plt.axis('off')
            st.pyplot(fig_net)

        with tab_heatmap:
            fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='Purples')
            fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)
            
    else:
        st.info("연관성 분석을 수행하기에는 필터링된 데이터가 너무 적습니다. 조건 범위를 넓혀주세요.")

# ---------------------------------------------------------
# 5. 실시간 AI 팩트체크
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🕵️‍♀️ 실시간 AI 팩트체크 & 편향성 탐지")

tab1, tab2 = st.tabs(["📜 현재 필터링된 기사 목록에서 검증", "✍️ 직접 입력해서 검증"])
target_article = ""
target_url = "" 

with tab1:
    if not global_df_filtered.empty:
        top_articles = global_df_filtered.sort_values(by='일자', ascending=False).head(50)
        has_publisher = '언론사' in top_articles.columns
        
        url_col = None
        if 'URL' in top_articles.columns: 
            url_col = 'URL'
        elif '기사 URL' in top_articles.columns: 
            url_col = '기사 URL'
        
        display_dict = {}
        for _, row in top_articles.iterrows():
            date_str = row['일자'].strftime('%Y-%m-%d')
            publisher_str = row['언론사'] if has_publisher else "알수없음"
            title_str = str(row['제목'])
            
            url_str = str(row[url_col]) if url_col and pd.notna(row[url_col]) else ""
            
            display_text = f"[{date_str}] ({publisher_str}) {title_str}"
            display_dict[display_text] = {
                "title": title_str,
                "url": url_str
            }
        
        selected_option = st.selectbox(
            "검증할 기사를 선택하세요 (최신순 50건):", 
            list(display_dict.keys())
        )
        
        if selected_option: 
            target_article = display_dict[selected_option]["title"]
            target_url = display_dict[selected_option]["url"] 
    else:
        st.warning("조건에 맞는 기사가 없습니다.")

with tab2:
    input_text = st.text_area("의심되는 기사 제목이나 내용을 입력하세요:", height=100)
    if input_text: 
        target_article = input_text
        target_url = ""

if st.button("🔍 팩트체크 시작"):
    if target_article:
        with st.spinner("⏳ AI 분석 중..."):
            time.sleep(0.5)
            result = classifier(target_article)[0]
            label = "긍정" if result['label'] == 'LABEL_1' else "부정"
            score = round(result['score'] * 100, 2)
            
            if score >= 90:
                level_text = "매우 강함 (주의 요망)"
                social_guide = "기사에 자극적인 단어나 감정적 표현이 집중적으로 사용되었습니다. <b>사회적 편견을 조장하거나 과장된 어뷰징 기사</b>일 가능성이 높으므로, 타 언론사의 팩트체크가 강력히 권장됩니다."
            elif score >= 70:
                level_text = "뚜렷한 논조"
                social_guide = "기자의 주관이나 특정 시각이 뚜렷하게 반영된 기사입니다. 객관적 사실과 의견을 분리하여 <b>균형 잡힌 시각</b>으로 수용할 필요가 있습니다."
            else:
                level_text = "비교적 중립/객관적"
                social_guide = "감정적인 단어 사용이 적고, <b>사실(Fact) 전달 위주</b>로 건조하게 작성되었을 확률이 높습니다. 비교적 객관적인 정보로 받아들일 수 있습니다."
            
            col_title, col_btn = st.columns([4, 1])
            with col_title:
                st.markdown(f"**분석 대상:** {target_article}")
            with col_btn:
                if target_url and target_url.startswith("http"):
                    st.link_button("🔗 기사 원문 보기", target_url)
            
            if label == "부정":
                st.error(f"🚨 **[부정/비판 편향성]** AI 확신도: {score}% ({level_text})")
            else:
                st.success(f"✅ **[긍정/희망 편향성]** AI 확신도: {score}% ({level_text})")
                
            st.info(f"💡 **AI 판단 가이드 (Media Literacy):** \n\n"
                    f"단순히 기사의 내용이 '좋다/나쁘다'를 넘어, 이 기사가 언론의 객관성을 얼마나 유지하고 있는지를 보여주는 지표입니다. AI가 언어적 패턴을 분석한 결과, 이 기사는 **{label} 프레임**에 속합니다.\n\n"
                    f"**📌 상식적 해석:** {social_guide}")
    else:
        st.warning("기사를 선택하거나 입력해주세요.")
