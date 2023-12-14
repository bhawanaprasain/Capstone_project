from src.file_handler import read_json_file,parse_pdf,list_files_in_directory
from src.vector_store import ChromaDBHandler

json_file ='/home/fm-pc-lt-149/Documents/Capstone_project/data/faq.json'
faqs = read_json_file(json_file)
chroma_db_handler = ChromaDBHandler(collection_name="healthcare")
for index, row in enumerate(faqs):
    chroma_db_handler.add_documents(
        documents=[row["answer"]],
        metadatas=[{"source": json_file}],
        ids=[str(index)]
    )