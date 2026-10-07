import os
import time
from django.conf import settings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", transport="rest")

# Initialize Chroma DB
CHROMA_PATH = os.path.join(settings.BASE_DIR, 'data', 'chroma_db')
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Set up retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

# Create QA Prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create chain
rag_chain = prompt | llm | StrOutputParser()

def query_rag(query_text: str):
    # Fetch documents manually to have them for sources
    docs = retriever.invoke(query_text)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    
    # Generate answer
    answer = rag_chain.invoke({"input": query_text, "context": context_text})
    
    sources = []
    for doc in docs:
        sources.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
        
    return {
        "answer": answer,
        "sources": sources
    }

def ingest_single_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        return 0
    # Increase chunk size to create fewer chunks, meaning fewer API calls
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    # Let LangChain handle the batching internally (it optimizes payload size)
    # Use exponential backoff for rate limits instead of a flat 60 seconds
    retry_delay = 5
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            vector_store.add_documents(chunks)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff: 5s, 10s, 20s, 40s
            else:
                raise e
                
    return len(chunks)
