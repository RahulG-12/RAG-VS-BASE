import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_rag():
    df = pd.read_csv("data/dataset.csv")
    texts = df["answer"].tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    documents = splitter.create_documents(texts)

    embedding = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.from_documents(
        documents,
        embedding
    )

    return vectorstore


def rag_answer(vectorstore, query):
    llm = OllamaLLM(model="phi3:mini")

    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)

    context = " ".join([doc.page_content for doc in docs])

    prompt = f"""
You are a medical assistant AI.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response


def base_answer(query):
    llm = OllamaLLM(model="phi3:mini")
    return llm.invoke(query)