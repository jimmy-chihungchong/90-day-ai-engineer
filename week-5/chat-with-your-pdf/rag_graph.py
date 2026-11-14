"""Self-RAG graph for chat-with-your-PDF (logic extracted from the notebook, no Streamlit)."""

import os
import tempfile
from typing import List

from dotenv import load_dotenv
from typing_extensions import TypedDict

load_dotenv()
os.environ.setdefault("USER_AGENT", "chat-with-your-pdf")  # silences WebBaseLoader warning

from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MAX_RETRIES = 2  # max number of generation attempts before giving up

BASE_URL = os.environ["LMSTUDIO_BASE_URL"]
CHAT_MODEL = os.environ["LMSTUDIO_CHAT_MODEL"]
EMBED_MODEL = os.environ["LMSTUDIO_EMBED_MODEL"]


# ---------------------------------------------------------------- models
def get_llm():
    """Chat model served by LM Studio (OpenAI-compatible API)."""
    return ChatOpenAI(model=CHAT_MODEL, base_url=BASE_URL, api_key="lm-studio", temperature=0)


def get_embeddings():
    """Embedding model served by LM Studio."""
    return OpenAIEmbeddings(
        model=EMBED_MODEL,
        base_url=BASE_URL,
        api_key="lm-studio",
        check_embedding_ctx_length=False,  # LM Studio expects raw strings, not token arrays
    )


# ---------------------------------------------------------------- loading / retriever
def load_pdfs(pdfs):
    """pdfs: iterable of (filename, bytes)."""
    docs = []
    for name, data in pdfs:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(data)
        try:
            loaded = PyMuPDFLoader(f.name).load()
        finally:
            os.remove(f.name)
        for d in loaded:
            d.metadata["source"] = name  # temp path -> real filename
        docs.extend(loaded)
    return docs


def load_urls(urls):
    """Load web pages; returns (docs, failed_urls) so one bad URL doesn't kill indexing."""
    docs, failed = [], []
    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception:
            failed.append(url)
    return docs, failed


def build_retriever(pdfs=(), urls=()):
    """Index PDFs and web pages into one in-memory Chroma store.

    Returns (retriever, failed_urls).
    """
    web_docs, failed = load_urls(urls)
    docs = load_pdfs(pdfs) + web_docs
    if not docs:
        raise ValueError("No content could be loaded. Add a PDF or a reachable URL.")
    splits = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    ).split_documents(docs)
    store = Chroma.from_documents(splits, embedding=get_embeddings())
    return store.as_retriever(), failed


# ---------------------------------------------------------------- state
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retries: int


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------- graph
def build_app(retriever):
    """Build and compile the self-RAG graph around the given retriever."""
    llm = get_llm()

    # Retrieval grader
    retrieval_grader = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing relevance of a retrieved document to a user question. \n "
                    "It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n"
                    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
                    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.",
                ),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        | llm.with_structured_output(GradeDocuments)
    )

    # Generator
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
                "to answer the question. If you don't know the answer, just say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n"
                "Question: {question} \nContext: {context} \nAnswer:",
            )
        ]
    )
    rag_chain = rag_prompt | llm | StrOutputParser()

    # Hallucination grader
    hallucination_grader = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n "
                    "Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.",
                ),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        | llm.with_structured_output(GradeHallucinations)
    )

    # Answer grader
    answer_grader = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a grader assessing whether an answer addresses / resolves a question \n "
                    "Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question.",
                ),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )
        | llm.with_structured_output(GradeAnswer)
    )

    # Question re-writer
    question_rewriter = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You a question re-writer that converts an input question to a better version that is optimized \n "
                    "for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. "
                    "Reply with ONLY the improved question, no explanation.",
                ),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    # ---- nodes
    def retrieve(state):
        question = state["question"]
        return {"documents": retriever.invoke(question), "question": question}

    def generate(state):
        question = state["question"]
        documents = state["documents"]
        retries = state.get("retries", 0) + 1
        generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "retries": retries,
        }

    def grade_documents(state):
        question = state["question"]
        filtered = [
            d
            for d in state["documents"]
            if retrieval_grader.invoke({"question": question, "document": d.page_content}).binary_score
            == "yes"
        ]
        return {"documents": filtered, "question": question}

    def transform_query(state):
        better_question = question_rewriter.invoke({"question": state["question"]})
        return {"documents": state["documents"], "question": better_question}

    # ---- edges
    def decide_to_generate(state):
        return "generate" if state["documents"] else "transform_query"

    def grade_generation(state):
        retries = state.get("retries", 0)
        grounded = hallucination_grader.invoke(
            {"documents": format_docs(state["documents"]), "generation": state["generation"]}
        ).binary_score
        if grounded == "yes":
            useful = answer_grader.invoke(
                {"question": state["question"], "generation": state["generation"]}
            ).binary_score
            if useful == "yes":
                return "useful"
            return "give up" if retries >= MAX_RETRIES else "not useful"
        return "give up" if retries >= MAX_RETRIES else "not supported"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "transform_query",
            "give up": END,
        },
    )
    return workflow.compile()
