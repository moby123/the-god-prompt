from qdrant_client import QdrantClient
import os

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

for collection in ["gita", "bible", "quran"]:
    count = client.count(collection_name=collection, exact=True).count
    print(f"{collection}: {count} vectors")
