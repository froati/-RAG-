import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, List

# 같은 폴더에 있는 .env 로드
load_dotenv()

class RetrievalResponse(BaseModel):
    Reasoning: str = Field(description="검색의 필요유무를 추론하는 과정")
    Retrieve: Literal['Yes', 'No'] = Field(description="검색 필요유무")

class RelevanceResponse(BaseModel):
    Reasoning: str = Field(description="연관문서의 관련성 평가 추론과정")
    ISREL: Literal['Relevant', 'Irrelevant'] = Field(description="관련성 평가 결과")

class GenerationResponse(BaseModel):
    response: str = Field(description="생성된 답변")

class SupportResponse(BaseModel):
    Reasoning: str = Field(description="답변이 문서에 근거하는지 평가")
    ISSUP: Literal['Fully supported', 'Partially supported', 'No support'] = Field(description="지원 평가 결과")

class InvestmentRAGEngine:
    def __init__(self, pdf_paths: List[str], index_path: str = "faiss_index"):
        self.pdf_paths = pdf_paths
        self.index_path = os.path.join(os.path.dirname(__file__), index_path)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.vector_db = self._setup_vector_db()

    def _setup_vector_db(self):
        # 1. 이미 저장된 인덱스가 있는지 확인
        # index_path 폴더가 있고 그 안에 index.faiss 파일이 있는지 확인해야 함
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            print(f"기존 인덱스 로드 중: {self.index_path}")
            return FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True # 로컬 저장이므로 허용
            )

        # 2. 저장된 인덱스가 없으면 PDF 파싱 및 생성
        print("새로운 인덱스 생성 중 (PDF 분석)...")
        all_docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        for path in self.pdf_paths:
            if os.path.exists(path):
                loader = PyPDFLoader(path)
                docs = loader.load_and_split(text_splitter)
                all_docs.extend(docs)

        if all_docs:
            db = FAISS.from_documents(all_docs, self.embeddings)
            # 생성 후 로컬에 저장
            db.save_local(self.index_path)
            print(f"인덱스 저장 완료: {self.index_path}")
            return db
        return None


    def process_query(self, query: str):
        # 1. Retrieval Decision
        if not self.vector_db:
            return "분석할 문서가 없습니다.", []

        # 2. Retrieve (k값을 늘려 더 많은 컨텍스트 확보)
        docs = self.vector_db.similarity_search(query, k=5)
        context = "\n".join([f"[문서 {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
        
        # 3. Generate with OpenAI
        prompt = PromptTemplate.from_template("""
        당신은 삼성전자 투자 전문 비서입니다. 제공된 [컨텍스트]를 바탕으로 사용자의 질문에 답변하세요.
        
        [지시사항]
        - 재무 제표나 배당 관련 질문의 경우, 표 형식의 데이터에서 정확한 수치를 찾아 답변하세요.
        - 답변은 친절하고 전문적이어야 하며, 반드시 제공된 정보에만 근거하세요.
        - 만약 컨텍스트에 답변에 필요한 수치가 명확히 나와 있다면, "확인할 수 없습니다" 대신 해당 수치를 정확히 언급하세요.
        - 수치를 언급할 때는 단위를 포함하세요 (예: 372원, 2조 4천억원 등).
        
        질문: {query}
        [컨텍스트]
        {context}
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({"query": query, "context": context})
        
        sources = [{"title": doc.metadata.get('source', '알 수 없음'), "page": doc.metadata.get('page', 0)} for doc in docs]
        
        return response.content, sources

    def get_investment_strategy(self, news_list: List[dict]):
        """
        뉴스 데이터와 보고서 데이터를 종합하여 투자 전략을 생성합니다.
        """
        if not self.vector_db:
            return "분석할 문서가 없습니다."

        # 최신 재무 현황 파악을 위한 검색
        docs = self.vector_db.similarity_search("삼성전자 실적 및 향후 전망", k=3)
        report_context = "\n".join([doc.page_content for doc in docs])
        
        news_context = "\n".join([f"- {n['title']}: {n['summary']}" for n in news_list])

        prompt = PromptTemplate.from_template("""
        당신은 삼성전자 전문 투자 전략가입니다. 아래 제공된 [공시 보고서 정보]와 [최신 뉴스]를 바탕으로 투자 전략을 수립하세요.
        
        [공시 보고서 정보]
        {report_context}
        
        [최신 뉴스]
        {news_context}
        
        [지시사항]
        - 1개월(단기), 3개월(중기), 6개월(장기) 관점에서 각각 투자 전략을 작성하세요.
        - 각 기간별로 핵심 전략을 명확한 문장으로 설명하세요.
        - 보고서의 정밀한 수치와 뉴스의 최신 트렌드를 적절히 결합하세요.
        - 전문적이면서도 투자자가 이해하기 쉬운 톤을 유지하세요.
        
        [출력 형식]
        1. 1개월(단기) 전략: [전략 내용]
        2. 3개월(중기) 전략: [전략 내용]
        3. 6개월(장기) 전략: [전략 내용]
        """)
        
        chain = prompt | self.llm
        response = chain.invoke({
            "report_context": report_context,
            "news_context": news_context
        })
        
        return response.content
