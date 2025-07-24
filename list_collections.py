from qdrant_client import QdrantClient

# Replace this with your actual Qdrant Cloud URL and API key
client = QdrantClient(
    url="https://dacad0f1-ddd8-48c9-959c-eb538a80a1ca.europe-west3-0.gcp.cloud.qdrant.io",   api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.F4UGDcUEWsW98rTkTZkD8zvewZjlcKMvTsPQrvoDPdA"
)

print("🔍 Existing Collections:")
print(client.get_collections())
