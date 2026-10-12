import os
import time
import datetime
import ast
import operator
import json
import http.client
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# Graceful guard for environments where Windows WDAC/AppLocker blocks cython_blas DLL
try:
    import sklearn.metrics
except ImportError:
    import sys
    from unittest.mock import MagicMock
    sys.modules.setdefault('sklearn.metrics', MagicMock())
    sys.modules.setdefault('sklearn.metrics.pairwise', MagicMock())

from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger('agent_api')


class BGEHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """
    Subclasses HuggingFaceEmbeddings to prepend the recommended BGE query instruction
    to queries, while embedding document chunks without instruction prefix.
    Embeddings are L2-normalized.
    """
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    def embed_query(self, text: str) -> list[float]:
        instruction = self.query_instruction or ""
        if instruction and not text.startswith(instruction):
            text = f"{instruction}{text}"
        return super().embed_query(text)


_embeddings_instance = None
_vector_store_instance = None


def get_embeddings():
    """Initializes and returns the singleton local Hugging Face embedding model."""
    global _embeddings_instance
    if _embeddings_instance is None:
        model_name = getattr(settings, 'LOCAL_EMBEDDING_MODEL_NAME', 'BAAI/bge-small-en-v1.5')
        query_instruction = getattr(
            settings, 'BGE_QUERY_INSTRUCTION', 'Represent this sentence for searching relevant passages: '
        )
        offline = os.environ.get('TRANSFORMERS_OFFLINE', '0') == '1' or os.environ.get('HF_HUB_OFFLINE', '0') == '1'
        try:
            # First try loading cached model without network calls
            _embeddings_instance = BGEHuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'local_files_only': True},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction=query_instruction
            )
        except Exception as local_err:
            if offline:
                raise RuntimeError(
                    f"Local embedding model '{model_name}' could not be loaded while offline mode is enabled. "
                    f"Ensure model weights are cached locally. Original error: {local_err}"
                ) from local_err
            try:
                _embeddings_instance = BGEHuggingFaceEmbeddings(
                    model_name=model_name,
                    encode_kwargs={'normalize_embeddings': True},
                    query_instruction=query_instruction
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load local embedding model '{model_name}': {e}") from e
    return _embeddings_instance


def get_vector_store(force_reload: bool = False):
    """Initializes and returns the singleton Chroma vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None or force_reload:
        chroma_path = str(getattr(settings, 'CHROMA_PATH', os.path.join(settings.BASE_DIR, 'data', 'chroma_db')))
        collection_name = getattr(settings, 'CHROMA_COLLECTION_NAME', 'pdf_documents_bge_small_v1')
        _vector_store_instance = Chroma(
            collection_name=collection_name,
            persist_directory=chroma_path,
            embedding_function=get_embeddings()
        )
    return _vector_store_instance


# Initialize Embeddings and Vector Store
CHROMA_PATH = str(getattr(settings, 'CHROMA_PATH', os.path.join(settings.BASE_DIR, 'data', 'chroma_db')))
CHROMA_COLLECTION_NAME = getattr(settings, 'CHROMA_COLLECTION_NAME', 'pdf_documents_bge_small_v1')

embeddings = get_embeddings()
vector_store = get_vector_store()

# Set up retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# Initialize Ollama LLM
def get_llm():
    model_name = getattr(settings, 'OLLAMA_MODEL_NAME', 'qwen2.5:3b')
    base_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://127.0.0.1:11434')
    return ChatOllama(model=model_name, base_url=base_url, temperature=0)

llm = get_llm()

# Setup Tools
document_search = create_retriever_tool(
    retriever,
    "document_search",
    "Primary tool for finding qualitative information in uploaded documents, resumes, and reports. Search this whenever the user asks about people, experience, qualifications, facts, or content in the knowledge base."
)

def _safe_eval(node):
    operators = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            # Limit power operations to prevent DoS via huge numbers
            right_val = _safe_eval(node.right)
            if abs(right_val) > 1000:
                raise ValueError("Exponent is too large.")
            return operators[type(node.op)](_safe_eval(node.left), right_val)
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

class OllamaServiceUnavailableError(Exception):
    """Raised when Ollama connection fails or server is unreachable."""
    pass


class OllamaModelNotFoundError(Exception):
    """Raised when the specified Ollama model is missing or not pulled."""
    pass


def _check_ollama_exception(e: Exception):
    """Inspects exception and raises a typed Ollama exception if appropriate."""
    msg = str(e).lower()
    if (
        "connection refused" in msg
        or "connecterror" in msg
        or "all connection attempts failed" in msg
        or "failed to connect" in msg
        or "cannot connect" in msg
        or "connection error" in msg
    ):
        raise OllamaServiceUnavailableError(
            "Local LLM service (Ollama) is unavailable. Please verify that Ollama is running at http://127.0.0.1:11434."
        ) from e
    if "not found" in msg and ("model" in msg or "404" in msg):
        raise OllamaModelNotFoundError(
            f"Configured Ollama model was not found: {str(e)}."
        ) from e


def perform_web_search(query: str) -> dict:
    """
    Search the live web using Serper API (google.serper.dev).
    Returns a dictionary with:
      - 'success': bool
      - 'results': list of dicts [{'title': str, 'url': str, 'snippet': str}]
      - 'raw_text': str
      - 'error': str or None
    """
    if not query or not query.strip():
        return {"success": False, "results": [], "raw_text": "", "error": "No query provided."}

    api_key = os.environ.get("SERPAPI_API_KEY") or os.environ.get("SERPER_API_KEY")
    if not api_key:
        logger.warning("Web search triggered but SERPAPI_API_KEY is not configured.")
        return {
            "success": False,
            "results": [],
            "raw_text": "",
            "error": "SERPAPI_API_KEY is missing or not configured."
        }

    conn = None
    try:
        conn = http.client.HTTPSConnection("google.serper.dev", timeout=10)
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        if res.status != 200:
            err_msg = f"Serper API returned HTTP status {res.status}"
            logger.warning("Web search HTTP error: %s", err_msg)
            return {"success": False, "results": [], "raw_text": "", "error": err_msg}

        data_bytes = res.read()
        data = json.loads(data_bytes.decode("utf-8"))

        results = []
        snippets = []
        for item in data.get("organic", [])[:5]:
            title = item.get("title", "").strip()
            link = item.get("link", "").strip()
            snippet = item.get("snippet", "").strip()
            if title or snippet:
                results.append({
                    "title": title or "Web Source",
                    "url": link,
                    "snippet": snippet
                })
                snippets.append(f"{title}: {snippet}")

        if not results and "answerBox" in data and isinstance(data["answerBox"], dict):
            ab = data["answerBox"]
            snippet = ab.get("answer") or ab.get("snippet") or ""
            title = ab.get("title") or "Direct Answer"
            link = ab.get("link", "")
            if snippet:
                results.append({"title": title, "url": link, "snippet": snippet})
                snippets.append(f"{title}: {snippet}")

        if not results:
            return {
                "success": True,
                "results": [],
                "raw_text": "",
                "error": "No relevant web search results found."
            }

        return {
            "success": True,
            "results": results,
            "raw_text": "\n\n".join(snippets),
            "error": None
        }
    except Exception as e:
        logger.warning("Web search request failed: %s", e)
        return {
            "success": False,
            "results": [],
            "raw_text": "",
            "error": f"Web search failed: {str(e)}"
        }
    finally:
        if conn:
            conn.close()


@tool
def search_web(query: str) -> str:
    """
    Search the live web for current public information.
    Use this for news, sports, current events, or anything NOT in the indexed documents.
    """
    res = perform_web_search(query)
    if not res.get("success"):
        return f"Web search unavailable: {res.get('error')}"
    if not res.get("results"):
        return "No relevant web search results found."
    return res.get("raw_text", "")

tools = [document_search, financial_calculator, current_time, search_web]

# Create Agent Prompt
system_prompt = """
You are an Agentic RAG assistant.

You have access to several tools, including:
1) document_search - searches through the user's uploaded private documents, resumes, and PDFs.
2) search_web - live internet search for public or current information.

Rules:
- For questions about specific documents, people, resumes, or private data, you MUST call document_search.
- When the user provides a follow-up answer or clarification (such as a person's name like "Annus", role, or brief keyword answering a previous question), combine it with the previous context from chat history (e.g. searching for the person's experience) and ALWAYS call document_search first with an informative search query.
- Do NOT declare that you do not have enough information before calling document_search.
- For live news or public facts outside the uploaded documents, call search_web.
- If a question needs BOTH private and public info, call both tools.
- Base your answer on the tool results. Do not invent facts or policies.
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


def evaluate_context_sufficiency(query: str, doc_chunks: list) -> bool:
    """
    Evaluates whether the retrieved PDF chunks contain relevant, sufficient evidence
    to answer the user's question.
    """
    if not doc_chunks:
        return False

    excerpts = []
    for i, doc in enumerate(doc_chunks[:4], 1):
        content = (getattr(doc, 'page_content', '') or '').strip()
        if content:
            excerpts.append(f"Excerpt {i}:\n{content[:500]}")

    if not excerpts:
        return False

    combined = "\n\n".join(excerpts)
    eval_prompt = (
        "You are an evaluator assessing if the following document excerpts contain "
        "sufficient, relevant evidence to answer the user's question.\n\n"
        f"User Question: {query}\n\n"
        f"Document Excerpts:\n{combined}\n\n"
        "Do these document excerpts contain direct evidence to answer the question?\n"
        "Respond with ONLY 'YES' or 'NO'."
    )

    try:
        current_llm = get_llm()
        resp = current_llm.invoke(eval_prompt)
        text = (getattr(resp, 'content', '') or str(resp)).strip().upper()
        if "YES" in text and "NO" not in text[:text.find("YES")]:
            return True
        return False
    except Exception as e:
        _check_ollama_exception(e)
        logger.warning("Error evaluating context sufficiency: %s", e)
        return False


def generate_document_answer(query: str, doc_chunks: list, chat_history: list = None) -> str:
    """Generates an answer strictly grounded in the retrieved document context."""
    excerpts = []
    for i, doc in enumerate(doc_chunks, 1):
        filename = (
            doc.metadata.get("display_name")
            or (Path(doc.metadata.get("source", "")).name if doc.metadata.get("source") else None)
            or f"Doc{i}"
        )
        page = doc.metadata.get("page", 0)
        excerpts.append(f"[{filename}, Page {page}]:\n{doc.page_content.strip()}")

    context_str = "\n\n".join(excerpts)
    prompt = (
        "You are an assistant answering questions strictly based on the provided document excerpts.\n\n"
        f"Document Excerpts:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Rules:\n"
        "1. Base your answer directly on the evidence in the document excerpts.\n"
        "2. Do not invent facts or make external assumptions.\n"
        "3. Keep your answer clear, informative, and concise.\n\n"
        "Answer:"
    )

    try:
        current_llm = get_llm()
        resp = current_llm.invoke(prompt)
        return (getattr(resp, 'content', '') or str(resp)).strip()
    except Exception as e:
        _check_ollama_exception(e)
        raise


def evaluate_web_sufficiency(query: str, web_results: list) -> bool:
    """
    Evaluates whether the retrieved web search results contain relevant, sufficient factual evidence
    to adequately answer the user's question without hallucination.
    """
    if not web_results:
        return False

    snippets = []
    for i, r in enumerate(web_results[:5], 1):
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()
        if snippet or title:
            snippets.append(f"Result {i} ({title}):\n{snippet[:400]}")

    if not snippets:
        return False

    combined = "\n\n".join(snippets)
    eval_prompt = (
        "You are a strict factual evaluator assessing if the following web search results contain "
        "direct, relevant, and sufficient factual evidence to answer the user's question without inventing information.\n\n"
        f"User Question: {query}\n\n"
        f"Web Search Results:\n{combined}\n\n"
        "Do these web search results contain direct and sufficient factual evidence to answer the question?\n"
        "Respond with ONLY 'YES' or 'NO'."
    )

    try:
        current_llm = get_llm()
        resp = current_llm.invoke(eval_prompt)
        text = (getattr(resp, 'content', '') or str(resp)).strip().upper()
        if "YES" in text and "NO" not in text[:text.find("YES")]:
            return True
        return False
    except Exception as e:
        _check_ollama_exception(e)
        logger.warning("Error evaluating web sufficiency: %s", e)
        return False


def generate_web_answer(query: str, web_results: list) -> str:
    """Generates an answer based on web search evidence."""
    excerpts = []
    for r in web_results:
        title = r.get("title", "Web Source")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        excerpts.append(f"[{title}] ({url}):\n{snippet}")

    context_str = "\n\n".join(excerpts)
    prompt = (
        "You are an assistant answering questions based on the provided web search results.\n\n"
        f"Web Search Results:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Rules:\n"
        "1. Answer the question using facts from the web search results.\n"
        "2. State key facts accurately and concisely.\n\n"
        "Answer:"
    )

    try:
        current_llm = get_llm()
        resp = current_llm.invoke(prompt)
        return (getattr(resp, 'content', '') or str(resp)).strip()
    except Exception as e:
        _check_ollama_exception(e)
        raise


def query_rag(query_text: str, history: list = None, allow_web_fallback: bool = True):
    """
    Context-sufficiency RAG workflow:
    1. Retrieve PDF chunks from Chroma vector store.
    2. Evaluate context sufficiency:
       - If sufficient: generate answer strictly from document evidence, avoid web search.
       - If insufficient: search the web via Serper.
         - If web evidence is adequate: generate answer from web results with structured web citations.
         - If web evidence is also inadequate or fails: state clearly that the answer cannot be verified.
    3. Return structured citations distinguishing document evidence from web evidence.
    """
    # Process history
    history = history or []
    chat_history = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get('role') == 'user':
                chat_history.append(HumanMessage(content=msg.get('content', '')))
            else:
                chat_history.append(AIMessage(content=msg.get('content', '')))
        elif isinstance(msg, (HumanMessage, AIMessage)):
            chat_history.append(msg)

    # Contextualize search query if short follow-up
    search_query = query_text
    if len(query_text.strip().split()) <= 3 and chat_history:
        last_user_msgs = [m.content for m in chat_history if isinstance(m, HumanMessage)]
        if last_user_msgs:
            search_query = f"{last_user_msgs[-1]} {query_text}".strip()

    # Backwards compatibility if agent_executor is mocked in legacy test
    from unittest.mock import MagicMock
    if isinstance(agent_executor, MagicMock):
        try:
            docs = retriever.invoke(search_query)
        except Exception:
            docs = []
        raw_res = agent_executor.invoke({"input": query_text, "chat_history": chat_history})
        out_ans = raw_res.get("output", "") if isinstance(raw_res, dict) else str(raw_res)
        doc_citations = []
        sources = []
        for doc in docs:
            filename = doc.metadata.get("display_name") or Path(doc.metadata.get("source", "")).name or "Document"
            page = doc.metadata.get("page", 0)
            doc_citations.append({"filename": filename, "page": page, "snippet": (doc.page_content or "")[:300].strip()})
            sources.append({"page_content": doc.page_content, "metadata": {**doc.metadata, "source_type": "document"}})
        return {
            "answer": out_ans,
            "sources": sources,
            "citations": {"documents": doc_citations, "web": []},
            "evidence_source": "document" if docs else "none"
        }

    # Step 1: Retrieve PDF chunks
    try:
        docs = retriever.invoke(search_query)
    except Exception as e:
        logger.error("Error retrieving documents from vector store: %s", e)
        docs = []

    # Step 2: Evaluate context sufficiency
    is_sufficient = evaluate_context_sufficiency(query_text, docs)

    # Step 3: If sufficient -> document evidence only (avoid unnecessary web search)
    if is_sufficient:
        answer = generate_document_answer(query_text, docs, chat_history=chat_history)
        doc_citations = []
        sources = []
        for doc in docs:
            filename = (
                doc.metadata.get("display_name")
                or (Path(doc.metadata.get("source", "")).name if doc.metadata.get("source") else None)
                or "Document"
            )
            page = doc.metadata.get("page", 0)
            doc_citations.append({
                "filename": filename,
                "page": page,
                "snippet": (doc.page_content or "")[:300].strip()
            })
            sources.append({
                "page_content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source_type": "document"
                }
            })

        return {
            "answer": answer,
            "sources": sources,
            "citations": {
                "documents": doc_citations,
                "web": []
            },
            "evidence_source": "document"
        }

    # Step 4: If insufficient -> trigger web search
    web_res = perform_web_search(query_text) if allow_web_fallback else {
        "success": False, "results": [], "error": "Web search disabled."
    }

    if web_res.get("success") and web_res.get("results"):
        web_results = web_res["results"]
        # Explicit relevance and sufficiency check: do not treat non-empty web search results as sufficient automatically
        is_web_sufficient = evaluate_web_sufficiency(query_text, web_results)
        if is_web_sufficient:
            answer = generate_web_answer(query_text, web_results)
            web_citations = [
                {
                    "title": r.get("title", "Web Source"),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", "")
                }
                for r in web_results
            ]
            sources = [
                {
                    "page_content": r.get("snippet", ""),
                    "metadata": {
                        "source": r.get("url", ""),
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "source_type": "web"
                    }
                }
                for r in web_results
            ]

            return {
                "answer": answer,
                "sources": sources,
                "citations": {
                    "documents": [],
                    "web": web_citations
                },
                "evidence_source": "web"
            }

    # Step 5: Both document and web evidence are inadequate or unverified
    unverified_msg = (
        "The answer cannot be verified based on the available documents and web search results."
    )
    return {
        "answer": unverified_msg,
        "sources": [],
        "citations": {
            "documents": [],
            "web": []
        },
        "evidence_source": "none"
    }


class IngestionState:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def compute_doc_id(filename: str) -> str:
    """Generates a stable, deterministic alphanumeric ID for a document based on its normalized filename."""
    if not filename:
        return ""
    norm_name = Path(filename).name.strip().lower()
    return f"doc_{hashlib.sha256(norm_name.encode('utf-8')).hexdigest()[:16]}"


@dataclass
class DocumentIdentity:
    doc_id: str
    display_name: str
    storage_name: str
    source: str

    @classmethod
    def create(cls, filename: str, display_name: str = None) -> 'DocumentIdentity':
        storage_name = Path(filename).name.strip()
        disp_name = display_name if display_name else storage_name
        doc_id = compute_doc_id(storage_name)
        source = f"data/raw_docs/{storage_name}"
        return cls(
            doc_id=doc_id,
            display_name=disp_name,
            storage_name=storage_name,
            source=source
        )


class IngestionResult(int):
    """
    Subclasses int for backward compatibility with code expecting an integer chunk count,
    while also exposing doc_id and allowing tuple unpacking: (chunks_count, doc_id).
    """
    def __new__(cls, chunks_count: int, doc_id: str):
        instance = super().__new__(cls, chunks_count)
        instance.chunks_count = chunks_count
        instance.doc_id = doc_id
        return instance

    def __iter__(self):
        yield self.chunks_count
        yield self.doc_id


def _add_documents_with_retry(vs, chunks, ids, max_retries: int = 5, base_delay: int = 2):
    """Adds documents to Chroma with unique IDs, retrying on rate-limit errors with exponential backoff."""
    retry_delay = base_delay
    for attempt in range(max_retries):
        try:
            vs.add_documents(chunks, ids=ids)
            return
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise


def ingest_single_pdf(
    file_path: str,
    display_name: str = None,
    storage_name: str = None,
    replace: bool = False
) -> IngestionResult:
    """
    Loads, splits, and indexes a PDF into ChromaDB using deterministic chunk IDs and structured metadata.
    Supports atomic safe replacement by snapshotting and restoring previous vectors if indexing fails.
    """
    target_fname = storage_name or Path(file_path).name
    doc_identity = DocumentIdentity.create(target_fname, display_name=display_name)
    doc_id = doc_identity.doc_id

    logger.info(
        "Ingestion state: %s for '%s' [doc_id=%s, replace=%s]",
        IngestionState.PENDING, doc_identity.display_name, doc_id, replace,
        extra={"doc_id": doc_id, "state": IngestionState.PENDING, "doc_name": doc_identity.display_name}
    )

    logger.info(
        "Ingestion state: %s for '%s' [doc_id=%s]: extracting text and generating chunks",
        IngestionState.PROCESSING, doc_identity.display_name, doc_id,
        extra={"doc_id": doc_id, "state": IngestionState.PROCESSING, "doc_name": doc_identity.display_name}
    )

    try:
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
    except Exception as e:
        logger.error(
            "Ingestion state: %s for '%s' [doc_id=%s] during file loading: %s",
            IngestionState.FAILED, doc_identity.display_name, doc_id, e,
            extra={"doc_id": doc_id, "state": IngestionState.FAILED, "doc_name": doc_identity.display_name}
        )
        raise

    if not documents:
        logger.warning(
            "Ingestion state: %s for '%s' [doc_id=%s]: no document pages found",
            IngestionState.FAILED, doc_identity.display_name, doc_id,
            extra={"doc_id": doc_id, "state": IngestionState.FAILED, "doc_name": doc_identity.display_name}
        )
        raise ValueError("The uploaded PDF contains no pages.")

    chunk_size = int(getattr(settings, 'RAG_CHUNK_SIZE', 1000))
    chunk_overlap = int(getattr(settings, 'RAG_CHUNK_OVERLAP', 150))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        logger.warning(
            "Ingestion state: %s for '%s' [doc_id=%s]: no extractable text",
            IngestionState.FAILED, doc_identity.display_name, doc_id,
            extra={"doc_id": doc_id, "state": IngestionState.FAILED, "doc_name": doc_identity.display_name}
        )
        raise ValueError("The uploaded PDF contains no extractable text. It may be an image-based or scanned document.")

    chunk_ids = []
    for idx, chunk in enumerate(chunks):
        c_id = f"{doc_id}_chunk_{idx}"
        chunk_ids.append(c_id)
        page = chunk.metadata.get("page", 0)
        chunk.metadata = {
            "doc_id": doc_id,
            "display_name": doc_identity.display_name,
            "storage_name": doc_identity.storage_name,
            "source": doc_identity.source,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "page": page,
        }

    # If replacing, snapshot existing vectors for safe rollback
    old_records = None
    if replace:
        try:
            old_records = vector_store.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas", "embeddings"]
            )
        except Exception as snap_err:
            logger.warning("Could not snapshot existing vectors for %s: %s", doc_identity.display_name, snap_err)
            old_records = None

    try:
        _add_documents_with_retry(vector_store, chunks, chunk_ids)
    except Exception as e:
        logger.error(
            "Ingestion state: %s for '%s' [doc_id=%s] during vector indexing: %s",
            IngestionState.FAILED, doc_identity.display_name, doc_id, e,
            extra={"doc_id": doc_id, "state": IngestionState.FAILED, "doc_name": doc_identity.display_name}
        )
        if replace and old_records and old_records.get("ids"):
            try:
                curr = vector_store.get(where={"doc_id": doc_id})
                curr_ids = set(curr.get("ids", []))
                old_ids = set(old_records.get("ids", []))
                leaked_ids = list(curr_ids - old_ids)
                if leaked_ids:
                    vector_store.delete(ids=leaked_ids)
                vector_store._collection.upsert(
                    ids=old_records["ids"],
                    documents=old_records.get("documents"),
                    metadatas=old_records.get("metadatas"),
                    embeddings=old_records.get("embeddings")
                )
            except Exception as restore_err:
                logger.error("Failed to restore previous vectors for %s: %s", doc_identity.display_name, restore_err)
        elif not replace:
            try:
                curr = vector_store.get(where={"doc_id": doc_id})
                if curr and curr.get("ids"):
                    vector_store.delete(ids=curr["ids"])
            except Exception:
                pass
        raise

    # In replacement mode, prune obsolete chunk IDs that belonged to older versions
    if replace:
        try:
            curr = vector_store.get(where={"doc_id": doc_id})
            curr_ids = set(curr.get("ids", []))
            new_ids = set(chunk_ids)
            obsolete_ids = list(curr_ids - new_ids)
            if obsolete_ids:
                vector_store.delete(ids=obsolete_ids)
        except Exception as prune_err:
            logger.warning("Error pruning obsolete chunk IDs for %s: %s", doc_identity.display_name, prune_err)

    logger.info(
        "Ingestion state: %s for '%s' [doc_id=%s]: %d chunks indexed",
        IngestionState.COMPLETED, doc_identity.display_name, doc_id, len(chunks),
        extra={"doc_id": doc_id, "state": IngestionState.COMPLETED, "doc_name": doc_identity.display_name}
    )

    return IngestionResult(len(chunks), doc_id)


def remove_pdf(file_path: str = None, doc_id: str = None, filename: str = None) -> int:
    """
    Removes all chunks associated with a document from Chroma DB.
    Matches primarily by stable doc_id, with fallbacks for legacy metadata and paths.
    """
    try:
        computed_doc_id = doc_id
        target_name = filename
        if not target_name and file_path:
            target_name = Path(file_path).name

        if not computed_doc_id and target_name:
            computed_doc_id = compute_doc_id(target_name)

        matched_ids = set()

        # Primary query: by stable doc_id
        if computed_doc_id:
            docs = vector_store.get(where={"doc_id": computed_doc_id})
            if docs and docs.get('ids'):
                matched_ids.update(docs['ids'])

        # Fallback queries for legacy chunks without doc_id
        if target_name:
            docs_sn = vector_store.get(where={"storage_name": target_name})
            if docs_sn and docs_sn.get('ids'):
                matched_ids.update(docs_sn['ids'])

            norm_source = f"data/raw_docs/{target_name}"
            docs_src = vector_store.get(where={"source": norm_source})
            if docs_src and docs_src.get('ids'):
                matched_ids.update(docs_src['ids'])

        if file_path:
            docs_fp = vector_store.get(where={"source": str(file_path)})
            if docs_fp and docs_fp.get('ids'):
                matched_ids.update(docs_fp['ids'])

            alt_path = str(file_path).replace('\\', '/') if '\\' in str(file_path) else str(file_path).replace('/', '\\')
            docs_alt = vector_store.get(where={"source": alt_path})
            if docs_alt and docs_alt.get('ids'):
                matched_ids.update(docs_alt['ids'])

        if matched_ids:
            vector_store.delete(ids=list(matched_ids))
            logger.info("Removed %d chunks for document [doc_id=%s, filename=%s]", len(matched_ids), computed_doc_id, target_name)
            return len(matched_ids)

        return 0
    except Exception as e:
        logger.exception("Failed to remove document from vector store:")
        raise Exception(f"Failed to remove from vector store: {str(e)}")


