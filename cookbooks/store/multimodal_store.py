from sentence_transformers import SentenceTransformer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient, models
from rich import print


from dotenv import load_dotenv
load_dotenv()

client = QdrantClient("http://localhost:6333")
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
image_encoder = SentenceTransformer("clip-ViT-B-32")

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com",
}


docs = [
    # --- Science & STEM ---
    {
        "id": 1,
        "text": "Einstein proposed that the laws of physics are the same for all observers, and the speed of light is constant.",
        "image_url": "./images/Albert_Einstein_Head.jpg",
        "metadata": {"topic": "physics", "subtopic": "relativity"}
    },
    {
        "id": 4,
        "text": "The solar system consists of the Sun and the celestial objects bound to it by gravity.",
        "image_url": "./images/Solar_sys8.jpg",
        "metadata": {"topic": "astronomy", "subtopic": "solar_system"}
    },
    {
        "id": 6,
        "text": "The Great Pyramid of Giza is the oldest of the Seven Wonders of the Ancient World.",
        "image_url": "./images/Kheops-Pyramid.jpg",
        "metadata": {"topic": "history", "subtopic": "ancient_egypt"}
    },
    # --- Humanities & Social Science ---
    {
        "id": 16,
        "text": "William Shakespeare wrote plays and sonnets that profoundly influenced the English language.",
        "image_url": "./images/Shakespeare.jpg",
        "metadata": {"topic": "literature", "subtopic": "english_poetry"}
    },
    {
        "id": 18,
        "text": "Beethoven was a German composer who bridged the Classical and Romantic eras in music.",
        "image_url": "./images/Beethoven.jpg",
        "metadata": {"topic": "music", "subtopic": "classical_composer"}
    }
]


def init_collection():
    client.recreate_collection(
        collection_name="text_multimodal",
        vectors_config={
            "text_vector": models.VectorParams(size=384, distance=models.Distance.COSINE),
            "image_vector": models.VectorParams(size=512, distance=models.Distance.COSINE)
        }
    )

def ingest_data(docs):
    points = []
    for doc in docs:
        img = Image.open(doc["image_url"])
        img_vec = image_encoder.encode([img])[0]
        text_vec = text_encoder.encode([doc["text"]])[0]

        points.append(models.PointStruct(
            id=doc["id"],
            vector={"text_vector": text_vec.tolist(), "image_vector": img_vec.tolist()},
            payload=doc["metadata"] | {"text": doc["text"], "image_url": doc["image_url"]}
        ))

    client.upsert(collection_name="text_multimodal", points=points)
    print("✅ 数据已导入 Qdrant")

def init_store():
    init_collection()
    ingest_data(docs)



if __name__ == "__main__":
    init_store()
    