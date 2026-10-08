import os
import time
import datetime
import ast
import operator
import json
import http.client
from django.conf import settings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", transport="rest")

# Initialize Chroma DB
CHROMA_PATH = os.path.join(settings.BASE_DIR, 'data', 'chroma_db')
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Set up retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# Setup Tools
document_search = create_retriever_tool(
    retriever,
    "doarch",
    "Primary tool for fcument_seinding qualitative information in financial reports."
)

def _safe_eval(node):
    operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                 ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_safe_eval(node.operand))
    raise TypeError("Unsupported mathematical operation.")

@tool
def financial_calculator(expression: str) -> str:
    """Accepts and evaluates basic mathematical expressions (like adding revenue or calculating margins)."""
    try:
        parsed = ast.parse(expression, mode='eval')
        return str(_safe_eval(parsed.body))
    except Exception as e:
        return f"Error evaluating expression: {e}"

@tool
def current_time(query: str = "") -> str:
    """Returns the current date and time. Useful when the user asks for things relative to 'now', 'today', 'this year', etc."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_web(query: str) -> str:
    """
    Search the live web for current public information.
    Use this for news, sports, current events, or anything NOT in the indexed documents.
    """
    if not query or not query.strip():
        return "No query provided."
    
    SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
    if not SERPAPI_API_KEY:
        return "SERPAPI_API_KEY is missing in .env. Cannot search the web."
    
    # Serper (google.serper.dev) is a Google Search API wrapper
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": SERPAPI_API_KEY,
        "Content-Type": "application/json",
    }
    try:
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        print(f"[Web] Search for '{query}' returned {len(data)} bytes")
        return data
    except Exception as e:
        return f"Web search failed: {str(e)}"

tools = [document_search, financial_calculator, current_time, search_web]

# Create Agent Prompt
system_prompt = """
You are an Agentic RAG assistant.

You have access to several tools, including:
1) document_search - searches through the user's uploaded private documents and PDFs.
2) search_web - live internet search for public or current information.

Rules:
- For questions about specific documents or private data, call document_search first.
- For live news or public facts outside the uploaded documents, call search_web.
- If a question needs BOTH private and public info, call both tools.
- Base your answer on the tool results. Do not invent facts or policies.
- If the tools return nothing useful and you are not confident, just say you do not have enough information.
- Keep answers clear and concise.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Initialize Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def query_rag(query_text: str, history: list = None):
    # Process history
    history = history or []
    chat_history = []
    for msg in history:
        if msg.get('role') == 'user':
            chat_history.append(HumanMessage(content=msg.get('content', '')))
        else:
            chat_history.append(AIMessage(content=msg.get('content', '')))
            
    # Fetch documents manually to have them for sources
    docs = retriever.invoke(query_text)
    
    # Generate answer using the agent
    response = agent_executor.invoke({
        "input": query_text,
        "chat_history": chat_history
    })
    
    raw_answer = response["output"]
    if isinstance(raw_answer, list):
        answer = "".join([part.get("text", "") for part in raw_answer if isinstance(part, dict) and "text" in part])
    else:
        answer = str(raw_answer)
    
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
    
    if not chunks:
        raise ValueError("The uploaded PDF contains no extractable text. It may be an image-based or scanned document.")
        
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

def remove_pdf(file_path: str):
    try:
        # Chroma's get method returns a dict with 'ids'
        docs = vector_store.get(where={"source": str(file_path)})
        if docs and docs.get('ids'):
            vector_store.delete(ids=docs['ids'])
            return len(docs['ids'])
        return 0
    except Exception as e:
        raise Exception(f"Failed to remove from vector store: {str(e)}")

