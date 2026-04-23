import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from langchain_classic.memory import ChatMessageHistory

# 설정 및 환경변수 로드
load_dotenv('data/.env')
api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader("data/2024_KB_부동산_보고서_최종.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def get_vectorstore(mode="chroma"):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    if mode == "chroma":
        persist_db = './chroma_db'
        if os.path.exists(persist_db):
            return Chroma(persist_directory=persist_db, embedding_function=embeddings)
        chunks = process_pdf()
        return Chroma.from_documents(chunks, embeddings, persist_directory=persist_db)
    else:
        faiss_db = './faiss_db'
        if os.path.exists(os.path.join(faiss_db, "index.faiss")):
            return FAISS.load_local(faiss_db, embeddings, allow_dangerous_deserialization=True)
        chunks = process_pdf()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_db)
        return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def limit_memory(input_dict):
    history = input_dict.get("chat_history", [])
    return history[-4:] if len(history) > 4 else history

@st.cache_resource
def initialize_chain():
    vectorstore = get_vectorstore(mode="faiss") # 또는 "faiss"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """당신은 KB 부동산 보고서 전문가입니다. 
    다음 컨텍스트를 바탕으로 답변하세요: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", template), ("placeholder", "{chat_history}"), ("human", "{question}")
    ])
    
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=api_key)

    base_chain = (
        RunnablePassthrough.assign(
            chat_history=limit_memory,
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt | model | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        base_chain, lambda s: ChatMessageHistory(),
        input_messages_key="question", history_messages_key="chat_history"
    )

def main():
    st.set_page_config(page_title="KB 부동산 보고서 챗봇")
    st.title("KB 부동산 보고서 AI 어드바이저")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        chain = initialize_chain()
        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                res = chain.invoke({"question": prompt}, 
                                  {"configurable": {"session_id": "st_session"}})
                st.markdown(res)
        st.session_state.messages.append({"role": "assistant", "content": res})

if __name__ == "__main__":
    main()
