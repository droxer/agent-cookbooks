from __future__ import annotations
from datetime import datetime, timezone
import json
from typing import Any, List, Optional, Iterable
from langgraph.store.base import BaseStore
# from langchain_community.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rich import print

from dotenv import load_dotenv
load_dotenv()


class QdrantStore(BaseStore):

    def __init__(
        self,
        collection_name: str = "langgraph_advanced_memory",
        host: str = "localhost",
        port: int = 6333,
        embedding_model: str = "all-MiniLM-L6-v2",
        distance = qmodels.Distance.COSINE,
    ):
        super().__init__()
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # 初始化 Collection
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=384, distance=qmodels.Distance.COSINE),
            )

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute multiple operations synchronously in a single batch."""
        # For simplicity, we'll execute operations sequentially
        # A more sophisticated implementation would batch them properly
        results = []
        for op in ops:
            # This is a simplified implementation
            # In a real implementation, you would handle different types of operations
            results.append(None)  # Placeholder result
        return results

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute multiple operations asynchronously in a single batch."""
        # For simplicity, we'll just call the sync version
        return self.batch(ops)

    def put(
        self,
        key: str,
        value: Any,
        session_id: str = "default",
        importance_score: float = 0.5,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        保存一个 state（自动加时间戳和重要度）
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        text = json.dumps(value, ensure_ascii=False)
        doc = Document(
            page_content=text,
            metadata={
                "key": key,
                "session_id": session_id,
                "timestamp": ts,
                "importance_score": float(importance_score),
            },
        )
        self.vectorstore.add_documents([doc])

    def get(self, key: str, session_id: str = "default") -> Optional[Any]:
        scroll, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.session_id", match=qmodels.MatchValue(value=session_id)
                    ),
                    qmodels.FieldCondition(key="metadata.key", match=qmodels.MatchValue(value=key)),
                ]
            ),
            limit=1,
        )
        if not scroll:
            return None
        text = scroll[0].payload.get("page_content", scroll[0].payload.get("text", ""))
        try:
            return json.loads(text)
        except Exception:
            return text

    def delete(self, key: str, session_id: str = "default") -> None:
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.session_id", match=qmodels.MatchValue(value=session_id)
                    ),
                    qmodels.FieldCondition(key="metadata.key", match=qmodels.MatchValue(value=key)),
                ]
            ),
        )
        if points:
            ids = [p.id for p in points]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=ids),
            )

    def search(self, query: str, session_id: str = "default", k: int = 5) -> List[Any]:
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.session_id",
                        match=qmodels.MatchValue(value=session_id)
                    )
                ]
            )
        )
        out = []
        for r in results:
            try:
                out.append(json.loads(r.page_content))
            except Exception:
                out.append(r.page_content)
        return out

    def weighted_search(
        self,
        query: str,
        session_id: str = "default",
        k: int = 10,
        decay_half_life_days: float = 7.0,
        importance_boost: float = 1.5,
    ) -> List[dict]:
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="metadata.session_id",
                        match=qmodels.MatchValue(value=session_id)
                    )
                ]
            )
        )

        scored_results = []
        now = datetime.now(timezone.utc).timestamp()

        for doc, sim_score in results:
            meta = doc.metadata or {}
            importance = float(meta.get("importance_score", 0.5))
            ts_str = meta.get("timestamp")
            ts = (
                datetime.fromisoformat(ts_str).timestamp()
                if ts_str
                else datetime.now(timezone.utc).timestamp()
            )

            # 时间衰减权重 (半衰期衰减)
            delta_days = (now - ts) / 86400
            time_weight = 0.5 ** (delta_days / decay_half_life_days)

            # 综合得分 = 语义相似度 × (importance^boost) × time_weight
            weighted_score = sim_score * (importance_boost * importance) * time_weight

            try:
                content = json.loads(doc.page_content)
            except Exception:
                content = doc.page_content

            scored_results.append(
                {
                    "score": weighted_score,
                    "importance": importance,
                    "recency_days": delta_days,
                    "item": content,
                }
            )

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results
 

if __name__ == "__main__":
    store = QdrantStore(collection_name="hybrid_advanced_store")

    store.put("mem1", {"summary": "User discussed AI education"}, session_id="userA", importance_score=0.9)
    store.put("mem2", {"summary": "User mentioned coffee preference"}, session_id="userA", importance_score=0.2)
    store.put("mem3", {"summary": "User talked about blockchain projects"}, session_id="userA", importance_score=0.8)

    print("\n Semantic Search：")
    for m in store.search("AI project", session_id="userA"):
        print(m)

    print("\n Weighted Semantic Search (Consider Time + Importance):")
    results = store.weighted_search("education AI", session_id="userA")
    for r in results[:3]:
        print(f"Score={r['score']:.4f} | imp={r['importance']} | age={r['recency_days']:.1f}d | item={r['item']}")
