"""
Quick RAG Demo: Manufacturing Knowledge Base
Ingests text files, embeds them, and answers questions using retrieved context.
"""
import os
import shutil

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"


# --- STEP 1: LOAD AND CHUNK ---
def load_documents():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "source": filename})
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc["content"])):
            chunks.append(
                {"content": text, "metadata": {"source": doc["source"], "chunk_id": i}}
            )
    return chunks


# --- STEP 2: EMBED AND STORE ---
def build_vectorstore(chunks):
    # Wipe old DB so reruns don't create duplicate chunks
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_texts(
        texts=[c["content"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )


# --- STEP 3: RETRIEVE AND ANSWER ---
PROMPT = ChatPromptTemplate.from_template(
    """You are a manufacturing expert assistant. Answer the question based ONLY on the provided context.
If the answer is not in the context, say "I don't have information about that in the documentation."

Context:
{context}

Question: {question}

Answer:"""
)


def ask_question(vectorstore, question: str, k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    relevant_docs = retriever.invoke(question)

    context = "\n\n---\n\n".join(d.page_content for d in relevant_docs)

    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return response.content, relevant_docs


# --- MAIN ---
if __name__ == "__main__":
    print("Loading documents...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} documents")

    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Embedding and storing in ChromaDB...")
    vectorstore = build_vectorstore(chunks)
    print(f"  Stored in {CHROMA_DIR}")

    print("\n" + "=" * 60)
    print("RAG SYSTEM READY")
    print("=" * 60)

    while True:
        question = input("\nYour question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue

        answer, docs = ask_question(vectorstore, question)
        print(f"\nAnswer:\n{answer}")
        print("\nSources used:")
        for d in docs:
            print(f"  - {d.metadata['source']} (chunk {d.metadata['chunk_id']})")