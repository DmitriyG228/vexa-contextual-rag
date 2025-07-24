import uuid
from typing import List, Dict, Any, Optional
from uuid import UUID
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,

    Filter,
    FieldCondition,

    SearchParams,
    TextIndexParams,
    TokenizerType,
    MatchAny,
    PointIdsList
)



import os



from voyageai import Client as VoyageClient


class QdrantSearchEngine:
    def __init__(self, voyage_api_key: str = None):
        qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        self.client = AsyncQdrantClient(qdrant_host, port=qdrant_port)
        
        # Get Voyage API key from environment if not provided
        if voyage_api_key is None:
            voyage_api_key = os.getenv('VOYAGE_API_KEY')
            if not voyage_api_key:
                raise ValueError("VOYAGE_API_KEY environment variable is required")
        
        self.voyage = VoyageClient(voyage_api_key)
        self.collection_name = "meeting_chunks"
        self.vector_size = 1024  # Voyage embedding size
        
    async def create_collection(self):
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        
        # Create payload indexes
        await self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name='content',
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                lowercase=True
            )
        )

    async def search(
        self,
        query: str,
        content_ids: Optional[List[str]] = None,
        k: int = 100
    ) -> List[Dict]:
        # Build filter
        filter_query = None
        if content_ids:
            filter_query = {
                "must": [
                    {"key": "content_id", "match": {"any": content_ids}}
                ]
            }
            
        # Get embeddings and extract values
        query_embedding = self.voyage.embed(query).embeddings[0]  # Get the first embedding's values
        
        # Perform search with updated filter
        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,  # Now passing the actual embedding values
            query_filter=filter_query,
            limit=k
        )
        return results

    async def drop_collection(self):
        try:
            await self.client.delete_collection(collection_name=self.collection_name)
            return True
        except Exception as e:

            return False

    async def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points from Qdrant collection by their IDs"""
        try:
            # Convert string IDs to UUIDs
            uuids = [UUID(id_str) for id_str in point_ids]
            
            # Delete points from collection
            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=uuids)
            )
            return True
        except Exception as e:
            print(f"Error deleting points from Qdrant: {e}")
            return False
