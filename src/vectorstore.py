from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
PERSIST_DIR = str(Path(__file__).parent / "chroma_db")

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