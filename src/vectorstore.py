from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
from src.config import PERSIST_DIR

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db_docs_store = Chroma(
    collection_name="db_docs",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

examples_store = Chroma(
    collection_name="query_examples",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)