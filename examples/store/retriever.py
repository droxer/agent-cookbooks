from sentence_transformers import SentenceTransformer
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client import models
from rich import print
import torch
from dotenv import load_dotenv
load_dotenv()

client = QdrantClient("http://localhost:6333")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("clip-ViT-B-32", device=device)


def retrieve_by_image(img_path: str):
    img = Image.open(img_path)
    img_vec = model.encode([img])[0]

    results = client.query_points(
        collection_name="text_multimodal",
        query=img_vec.tolist(),
        using="image_vector",
        limit=3
    )
    return results


def retrieve_by_text(text: str):
    text_vec = model.encode([text])[0]

    results = client.query_points(
        collection_name="text_multimodal",
        query=text_vec.tolist(),
        using="text_vector",
        limit=3
    )
    return results


def main():
    # Test image-to-image retrieval
    print("=== Image-to-Image Retrieval ===")
    results = retrieve_by_image("images/beethoven_white.jpeg")
    print(results)

    print("\n=== Text-to-Text Retrieval ===")
    results = retrieve_by_text("What did Einstein contribute to science?")
    print(results)


if __name__ == "__main__":
    main()