import os
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_classic.memory import ChatMessageHistory

# 시스템의 표준 출력을 utf-8로 강제 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# 환경 변수 로드
load_dotenv('data/.env')
api_key = os.getenv("OPENAI_API_KEY")


# PDF 처리 함수
@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 벡터 스토어 초기화
@st.cache_resource # 무거운 리소스(모델, vectorstore)를 한 번만 생성하고 이후에는 다시 만들지 않도록 하는 기능 - 계속 만들면 느려지기 때문에 효율화
def initialize_vectorstore():
    chunks = process_pdf()
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource # 무거운 리소스(모델, vectorstore)를 한 번만 생성하고 이후에는 다시 만들지 않도록 하는 기능 - 계속 만들면 느려지기 때문에 효율화
def load_vectorstore(folder_path):
    embeddings = OpenAIEmbeddings(openai_api_key = api_key)
    return FAISS.load_local(folder_path,embeddings, allow_dangerous_deserialization=True)

# 체인 초기화
@st.cache_resource
def initialize_chain():
    faiss_directory = './faiss_db'
    if os.path.isdir(faiss_directory):
        vectorstore = load_vectorstore(faiss_directory)
    else:
        vectorstore = initialize_vectorstore()
        vectorstore.save_local(faiss_directory)

    retriever = vectorstore.as_retriever(search_kwargs= {"k":3})

    template = """
    당신은 KB 부동산 보고서 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요.
    컨텍스트: {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def limit_memory(input_dict):
        history = input_dict.get("chat_history",[])
        return history[-4:] if len(history) > 4 else history

    base_chain = (
        # assign: RunnableWithMessageHistory를 사용하면, 
        # 내부적으로 저장소에 있는 전체 대화 기록을 불러와서 base_chain에 전달합니다.
        RunnablePassthrough.assign(
            chat_history = limit_memory,
            context = lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id: ChatMessageHistory(),
        input_messages_key="question",
        history_messages_key="chat_history",
    )

# Streamlit UI
def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇")
    st.title("KB 부동산 보고서 AI 어드바이저 yoon")
    st.caption("2024 KB 부동산 보고서 기반 질의응답 시스템")

    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 사용자 입력 처리
    if prompt := st.chat_input("부동산 관련 질문을 입력하세요"):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content":prompt})

        # 체인 초기화
        chain = initialize_chain()

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = chain.invoke(
                    {"question": prompt},
                    {"configurable": {"session_id": "streamlit_session"}}
                )
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

