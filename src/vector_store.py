import chromadb

class ChromaDBHandler:
    def __init__(self,collection_name):
        # Initialize the ChromaDB client
        self.client = chromadb.Client()

        # Get or create the collection named "test"
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents, metadatas, ids):
        # Add documents to the collection
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query_collection(self, query_texts, n_results):
        # Query the collection
        results = self.collection.query(query_texts=query_texts, n_results=n_results)
        return results