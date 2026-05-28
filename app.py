import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import os
import yfinance as yf
from rag_engine import InvestmentRAGEngine
from news_engine import get_samsung_news

# streamlit run app.py
# --- Page Config ---
st.set_page_config(
    page_title="삼성전자 스마트 투자 보조 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize RAG Engine ---
@st.cache_resource
def load_rag_engine():
    # 모든 파일이 현재 app.py와 같은 폴더에 있음
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    pdf_files = [
        os.path.join(current_dir, "[삼성전자]분기보고서(2026.05.15).pdf"),
        os.path.join(current_dir, "[삼성전자]사업보고서(2026.03.10).pdf")
    ]

    # 파일 존재 여부 체크 (터미널 출력)
    for f in pdf_files:
        if not os.path.exists(f):
            print(f"⚠️ 파일을 찾을 수 없음: {f}")

    return InvestmentRAGEngine(pdf_files, index_path="faiss_index")


try:
    rag_engine = load_rag_engine()
    if rag_engine and rag_engine.vector_db:
        st.sidebar.success(f"✅ 문서 학습 완료 ({len(rag_engine.pdf_paths)}개 파일)")
        for p in rag_engine.pdf_paths:
            st.sidebar.caption(f"📄 {os.path.basename(p)}")
    else:
        st.sidebar.warning("⚠️ 학습된 문서가 없습니다. PDF 파일을 확인해주세요.")
except Exception as e:
    st.error(f"RAG 엔진 로드 실패: {e}")
    rag_engine = None

# --- Custom CSS (Notion Style & Layout) ---
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .notion-text {
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        color: #37352f;
    }
    .source-accordion {
        background-color: #f7f6f3;
        border-radius: 5px;
        padding: 10px;
    }
    .badge-safe { background-color: #e2fceb; color: #216e39; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warning { background-color: #fff9db; color: #856404; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    .badge-danger { background-color: #ffe3e3; color: #cf222e; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
    
    /* 뉴스 카드 스타일 */
    .news-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .news-title {
        font-weight: bold;
        color: #1a73e8;
        text-decoration: none;
        font-size: 1.1em;
    }
    .news-meta {
        color: #70757a;
        font-size: 0.85em;
        margin-bottom: 5px;
    }
    
    /* 투자 전략 스타일 */
    .strategy-container {
        background-color: #f8f9fa;
        border-left: 5px solid #2ecc71;
        padding: 20px;
        border-radius: 0 8px 8px 0;
        margin-top: 20px;
    }
    .strategy-item {
        margin-bottom: 15px;
    }
    .strategy-label {
        font-weight: bold;
        color: #2c3e50;
        display: block;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_news_and_strategy():
    news = get_samsung_news(limit=3)
    strategy = rag_engine.get_investment_strategy(news) if rag_engine else "엔진이 로드되지 않았습니다."
    return news, strategy

@st.cache_data(ttl=3600)
def fetch_stock_data():
    ticker = "005930.KS"
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    info = stock.info
    metrics = {
        "CurrentPrice": info.get("currentPrice", df['Close'].iloc[-1] if not df.empty else 0),
        "PER": info.get("trailingPE", 0),
        "PBR": info.get("priceToBook", 0),
        "ROE": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
        "DividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0
    }
    return df, metrics

def create_stock_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA60'], name="MA60", line=dict(color='blue', width=1)))
    fig.update_layout(
        title="삼성전자 주가 추이 (최근 6개월)",
        yaxis_title="가격 (원)",
        yaxis=dict(tickformat=",.0f"), # 천 단위 콤마 추가
        xaxis_rangeslider_visible=False,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        template="plotly_white"
    )
    return fig

def create_gauge_chart(value, title, min_val, max_val):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.4], 'color': "#e2fceb"},
                {'range': [(max_val-min_val)*0.4, (max_val-min_val)*0.7], 'color': "#fff9db"},
                {'range': [(max_val-min_val)*0.7, max_val], 'color': "#ffe3e3"}
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --- Sidebar: Dashboard ---
with st.sidebar:
    st.header("📊 실시간 대시보드")
    
    try:
        stock_df, metrics = fetch_stock_data()
        
        diff = stock_df['Close'].iloc[-1] - stock_df['Close'].iloc[-2]
        pct = (diff / stock_df['Close'].iloc[-2]) * 100
        st.metric("현재가", f"{int(metrics['CurrentPrice']):,}원", f"{pct:.2f}%")
        
        st.subheader("재무 건전성")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**PER**: {metrics['PER']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PER"], "", 0, 30), width="stretch")
        with col2:
            st.markdown(f"**PBR**: {metrics['PBR']:.2f}")
            st.plotly_chart(create_gauge_chart(metrics["PBR"], "", 0, 3), width="stretch")
            
        st.subheader("시장 심리 (Sentiment)")
        sentiment_data = pd.DataFrame({
            "Sentiment": ["긍정", "중립", "부정"],
            "Ratio": [65, 20, 15]
        })
        fig_donut = px.pie(sentiment_data, values='Ratio', names='Sentiment', hole=.4,
                     color_discrete_sequence=['#2ecc71', '#95a5a6', '#e74c3c'])
        fig_donut.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_donut, width="stretch")
        
        st.subheader("시장 동향 점수")
        trend_data = pd.DataFrame({
            "Date": pd.date_range(start="2026-05-01", periods=10),
            "Score": [70, 72, 68, 75, 80, 78, 82, 85, 83, 88]
        })
        fig_line = px.line(trend_data, x="Date", y="Score")
        fig_line.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_line, width="stretch")
    except Exception as e:
        st.error(f"대시보드 로드 실패: {e}")

# --- Main Area: AI Assistant ---
st.title("🤖 삼성전자 AI 투자 비서")
st.markdown("---")

# 실시간 주가 차트 표시
if 'stock_df' in locals():
    st.plotly_chart(create_stock_chart(stock_df), width="stretch")
    st.markdown("<br>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="notion-text">{message["content"]}</div>', unsafe_allow_html=True)
        if "sources" in message:
            with st.expander("📚 출처 확인하기 (Source Accordion)"):
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['link']})")

# User Input
if prompt := st.chat_input("삼성전자의 최근 배당 정책에 대해 알려줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Self-RAG Process Visualization
        status_text = st.status("🔍 정보를 분석 중입니다...", expanded=True)
        time.sleep(0.5)
        status_text.write("1. 관련 문서 검색 중 (Knowledge Base: 삼성전자 보고서)")
        
        if rag_engine:
            response_text, sources = rag_engine.process_query(prompt)
            status_text.write("2. 답변 생성 및 자가 검증 중 (Gemini Self-Reflection)")
            time.sleep(0.5)
            status_text.update(label="✅ 분석 완료!", state="complete", expanded=False)
            
            # Display response
            message_placeholder.markdown(f'<div class="notion-text">{response_text}</div>', unsafe_allow_html=True)
            
            # Display sources
            if sources:
                with st.expander("📚 출처 확인하기 (Source Accordion)"):
                    for src in sources:
                        filename = os.path.basename(src['title'])
                        st.markdown(f"- **{filename}** (p.{src['page']})")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "sources": [{"title": os.path.basename(s['title']), "link": "#"} for s in sources]
            })
        else:
            status_text.update(label="❌ RAG 엔진 미로드", state="error", expanded=True)
            st.error("RAG 엔진이 설정되지 않았습니다. API 키와 파일 경로를 확인해주세요.")

# --- Bottom Section: News & Investment Strategy ---
st.markdown("---")

news_data, strategy_text = fetch_news_and_strategy()

# 1. 최근 주요 뉴스 (상단)
st.subheader("📰 최근 주요 뉴스")
for item in news_data:
    st.markdown(f"""
    <div class="news-card">
        <div class="news-meta">{item['press']}</div>
        <a href="{item['link']}" target="_blank" style="text-decoration: none;">
            <div class="news-title">{item['title']}</div>
        </a>
        <div class="notion-text" style="font-size: 0.9em; margin-top: 5px;">{item['summary']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2. AI 추천 투자 전략 (하단)
st.subheader("💡 AI 추천 투자 전략")
# 전략 텍스트 포맷팅 (개행 및 강조)
formatted_strategy = strategy_text.replace("1.", "### 1.").replace("2.", "### 2.").replace("3.", "### 3.")
st.markdown(f"""
<div class="strategy-container">
    <div class="notion-text">{formatted_strategy}</div>
</div>
""", unsafe_allow_html=True)
