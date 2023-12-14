from src.file_handler import read_json_file
from src.vector_store import ChromaDBHandler
from src.config import json_file

faqs = read_json_file(json_file)
chroma_db_handler = ChromaDBHandler(collection_name="healthcare")
for index, row in enumerate(faqs):
    chroma_db_handler.add_documents(
        documents=[row["answer"]],
        metadatas=[{"source": json_file}],
        ids=[str(index)]
    )