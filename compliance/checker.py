from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from compliance.retriever import get_retriever
from langchain_community.vectorstores import Chroma

def check_compliance(file_path):
    """
    Load a user file, embed it dynamically, and check compliance
    against the seeded policy vector DB.
    """

    loader = TextLoader(file_path, encoding = "utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    temp_vdb = Chroma.from_documents(
        documents=chunks,
        embedding = embeddings,
        persist_directory=None
    )
    project_retriever = temp_vdb.as_retriever(search_kwargs={"k":3})
    policy_retriever = get_retriever()
    results = []
    for chunk in chunks:
        relevant_rules = policy_retriever.get_relevant_documents(chunk.page_content)
        results.append({
            "text": chunk.page_content,
            "violations": [r.page_content for r in relevant_rules]
        })
    return results