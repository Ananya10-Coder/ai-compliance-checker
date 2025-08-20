from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

VECTORDB_DIR = "storage/chroma/policies"

def get_retriever():
    """Load the persisted Chroma DB and return a retriever object."""
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma(
        persist_directory=VECTORDB_DIR,
        embedding_function= embeddings
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    return retriever
