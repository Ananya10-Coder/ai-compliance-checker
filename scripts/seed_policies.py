from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

POLICY_DIR = "data/policies"
VECTORDB_DIR = "storage/chroma/policies"

def load_rules():
    docs = []
    for filename in os.listdir(POLICY_DIR):
        filepath = os.path.join(POLICY_DIR, filename)
        if(filename.endswith(".txt")):
            with open(filepath,"r") as f:
                for line in f:
                    if ":" in line:
                        rule_id, text = line.strip().split(":", 1)
                        docs.append(Document(page_content=text, metadata = {"id": rule_id}))
    return docs

def main():
    print("Loading compliance rules from data/policies/")
    docs = load_rules()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding= embeddings,
        persist_directory= VECTORDB_DIR
    )
    vectordb.persist()
    print(f"✅ Stored {len(docs)} compliance rules into {VECTORDB_DIR}")

if __name__ == "__main__":
    main()

