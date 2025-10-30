import numpy as np
from qdrant_client import QdrantClient
from rich import print

def verify_vector_consistency(
    client: QdrantClient,
    collection_name: str,
    vector_name: str = None,
    sample_size: int = 500,
    tolerance: float = 0.05,
    plot: bool = True
):
    print(f"🔍 Verifying vector normalization for [{collection_name}] ({vector_name or 'default'})...")

    count = client.count(collection_name).count
    total = min(count, sample_size)

    scroll_filter = None
    limit = total
    scroll_cursor = None
    vectors = []

    while len(vectors) < total:
        res, scroll_cursor = client.scroll(
            collection_name=collection_name,
            limit=min(100, total - len(vectors)),
            with_vectors=True,
            with_payload=False,
            scroll_filter=scroll_filter,
            offset=scroll_cursor
        )
        for point in res:
            v = point.vector[vector_name] if vector_name else point.vector
            vectors.append(np.array(v))
        if not scroll_cursor:
            break

    vectors = np.array(vectors)
    norms = np.linalg.norm(vectors, axis=1)

    mean_norm = np.mean(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    std_norm = np.std(norms)

    normalized_ratio = np.mean((norms > (1 - tolerance)) & (norms < (1 + tolerance)))

    result = {
        "sample_size": len(norms),
        "mean_norm": round(mean_norm, 4),
        "std_norm": round(std_norm, 4),
        "min_norm": round(min_norm, 4),
        "max_norm": round(max_norm, 4),
        "normalized_ratio": round(normalized_ratio, 4),
        "tolerance": tolerance,
    }

    print("\n📊 Vector Norm Statistics:")
    for k, v in result.items():
        print(f"{k:>18}: {v}")

    if normalized_ratio > 0.95:
        print("\n✅ Most vectors are normalized (>95%)")
    else:
        print("\n⚠️ Detected some vectors not normalized, please ensure consistency when inserting and querying!")

    return result


if __name__ == "__main__":
    client = QdrantClient("http://localhost:6333")

    verify_vector_consistency(
        client,
        collection_name="text_multimodal",
        vector_name="image_vector",  # 或 "text_vector"
        sample_size=500,
    )
