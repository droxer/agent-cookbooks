from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from PIL import Image
from qdrant_client import QdrantClient
from rich import print

from dotenv import load_dotenv
load_dotenv()

client = QdrantClient("http://localhost:6333")

image_encoder = SentenceTransformer("clip-ViT-B-32")


def retrieve_image_query(img_path: str):
    img = Image.open(img_path)
    img_vec = image_encoder.encode([img])[0]

    results = client.query_points(
        collection_name="text_multimodal",
        query=img_vec,
        using="image_vector",
        limit=3,
        score_threshold=0.6
    )
    return results


def main():
    # results = retrieve_image_query("images/Albert_Einstein.jpg")
    results = retrieve_image_query("images/beethoven_white.jpeg")
    print(results)

if __name__ == "__main__":
    main()