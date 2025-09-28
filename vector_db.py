from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    def __init__(self, url='http://localhost:6333', collection='docs', dim=3072):
        self.client = QdrantClient(url=url, timeout=30) # when no connection occurs within 30s, crash server
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    
    
    def upsert(self, ids, vectors, payloads):       # function to insert/update
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]   # required for qdrant  
        self.client.upsert(self.collection, points=points)

    
    def search(self, query_vector, top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        sources = set() # only stores unique sources. Order will get lost.

        for r in results:
            payload = getattr(r, 'payload', None) or {}
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.add(source)
        

        return {'contexts': contexts, 'sources': list(sources)}
