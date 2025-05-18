import os
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).parent.parent.parent

with open(BASE_DIR / "data.txt", "r") as f:
    lines = f.readlines()

# Create a DataFrame
df = pd.DataFrame({"Info": [line.strip() for line in lines]})

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# db_location = BASE_DIR/"langchain_db"
db_location = "./src/langchain_db"
add_doc = not os.path.exists(db_location)

if add_doc:
    documents = []
    ids = []

    for i, row in df.iterrows():
        # Create a new Document for each row
        doc = Document(page_content=row["Info"])
        documents.append(doc)
        ids.append(str(i))  # Add the ID for this document

try:
    vector_store = Chroma(
        collection_name="ebla_info",
        persist_directory=db_location,
        embedding_function=embeddings,
    )
    print(vector_store)
except Exception as e:
    print(e)

if add_doc:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
