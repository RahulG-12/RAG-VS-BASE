import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    def __init__(self, model_name):
        self.model_name = model_name
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        self.llm = OllamaLLM(model=model_name)
        self.vectorstore = None

    def build_vectorstore(self, dataset_path):
        df = pd.read_csv(dataset_path)
        texts = df["answer"].tolist()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        docs = splitter.create_documents(texts)

        self.vectorstore = FAISS.from_documents(
            docs,
            self.embedding
        )

    def rag_answer(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(query)
        context = " ".join([doc.page_content for doc in docs])

        prompt = f"""
Use the context to answer.

Context:
{context}

Question:
{query}
Answer:
"""
        return self.llm.invoke(prompt)

    def base_answer(self, query):
        return self.llm.invoke(query)