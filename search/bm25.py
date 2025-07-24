from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import List, Dict, Any, Optional
import os
import pandas as pd
from datetime import datetime
from elasticsearch import AsyncElasticsearch

class ElasticsearchBM25:
    def __init__(self, index_name: str = "meetings"):
        self.es_host = os.getenv('ELASTICSEARCH_HOST', 'localhost')
        self.es_port = os.getenv('ELASTICSEARCH_PORT', '9200')
        self.index_name = index_name
        self.es_client = None
        
    async def initialize(self):
        """Async initialization method"""
        try:
            # Configure client for Elasticsearch 8.x compatibility
            self.es_client = AsyncElasticsearch(
                f"http://{self.es_host}:{self.es_port}",
                request_timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )
            print(f"Connecting to Elasticsearch at {self.es_host}:{self.es_port}")
            await self.create_index()
            return self
        except Exception as e:
            print(f"Error initializing Elasticsearch connection: {str(e)}")
            # Close the client if it was created but initialization failed
            if self.es_client:
                try:
                    await self.es_client.close()
                except:
                    pass
                self.es_client = None
            return self


    @classmethod
    async def create(cls, index_name: str = "meetings"):
        """Factory method for creating and initializing instance"""
        instance = cls(index_name)
        await instance.initialize()
        return instance

    async def create_index(self):
        if not self.es_client:
            print("No Elasticsearch connection available")
            return
            
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {"type": "english"}
                    }
                },
                "similarity": {
                    "default": {"type": "BM25"}
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "content_id": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "meeting_id": {"type": "keyword"},
                    "formatted_time": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "chunk_index": {"type": "integer"},
                    "speakers": {"type": "keyword"},
                    "speaker": {"type": "keyword"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "content_type": {"type": "keyword"}
                }
            }
        }
        

        if not await self.es_client.indices.exists(index=self.index_name):
            await self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")
        else:
            print(f"Index {self.index_name} already exists")


    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        if not self.es_client:
            print("No Elasticsearch connection available")
            return False
            
        try:
            actions = [
                {
                    "_index": self.index_name,
                    "_source": {
                        "content_id": doc["content_id"],
                        "content": doc["content"],
                        "topic": doc["topic"],
                        "meeting_id": doc.get("meeting_id"),
                        "formatted_time": doc["formatted_time"],
                        "timestamp": doc["timestamp"],
                        "chunk_index": doc["chunk_index"],
                        "speaker": doc["speaker"],
                        "speakers": doc["speakers"],
                        "contextualized_content": doc["contextualized_content"],
                        "content_type": doc["content_type"]
                    }
                }
                for doc in documents
            ]
            success, _ = bulk(self.es_client, actions)
            self.es_client.indices.refresh(index=self.index_name)
            return success
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            return False

    async def search(
        self,
        query: str,
        content_ids: Optional[List[str]] = None,
        k: int = 100
    ) -> List[Dict]:
        # Build query
        should_clauses = [
            {"match": {"content": {"query": query, "boost": 1.0}}}
        ]
        
        # Add content filter if provided
        filter_clauses = []
        if content_ids:
            filter_clauses.append({"terms": {"content_id": content_ids}})
            
        # Build final query
        es_query = {
            "query": {
                "bool": {
                    "should": should_clauses,
                    "filter": filter_clauses,
                    "minimum_should_match": 1
                }
            },
            "size": k
        }
        
        results = await self.es_client.search(
            index=self.index_name,
            body=es_query
        )
        return results

    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from Elasticsearch by their IDs"""
        if not self.es_client:
            print("No Elasticsearch connection available")
            return False
            
        try:
            # Delete documents by ID
            for doc_id in document_ids:
                self.es_client.delete(
                    index=self.index_name,
                    id=doc_id,
                    ignore=[404]  # Ignore if document doesn't exist
                )
            
            # Refresh index to make changes visible
            self.es_client.indices.refresh(index=self.index_name)
            return True
        except Exception as e:
            print(f"Error deleting documents from Elasticsearch: {e}")
            return False

    async def close(self):
        """Close the Elasticsearch client connection"""
        if self.es_client:
            try:
                await self.es_client.close()
                print("Elasticsearch client connection closed")
            except Exception as e:
                print(f"Error closing Elasticsearch client: {e}")
            finally:
                self.es_client = None

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    def is_connected(self) -> bool:
        """Check if the Elasticsearch client is connected and ready"""
        return self.es_client is not None

    async def health_check(self) -> bool:
        """Check if Elasticsearch is healthy and accessible"""
        if not self.es_client:
            return False
        try:
            health = await self.es_client.cluster.health()
            return health.get('status') in ['green', 'yellow']
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

async def hybrid_search(
    query: str,
    qdrant_engine,
    es_engine: ElasticsearchBM25,
    meeting_ids: List[str] = None,
    speakers: List[str] = None,
    k: int = 10,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[Dict]:
    # Get results from both engines with speaker filtering
    semantic_results = await qdrant_engine.search(
        query, meeting_ids=meeting_ids, speakers=speakers, limit=k*2
    )
    bm25_results = es_engine.search(
        query, meeting_ids=meeting_ids, speakers=speakers, k=k*2
    )
    
    # Create dictionaries for scoring
    all_results = {}
    
    # Score semantic results
    for i, result in enumerate(semantic_results):
        key = (result['meeting_id'], result['formatted_time'])
        score = semantic_weight * (1.0 - (i / len(semantic_results)))
        all_results[key] = {
            'data': result,
            'score': score,
            'sources': ['semantic']
        }
    
    # Score BM25 results
    for i, result in enumerate(bm25_results):
        key = (result['meeting_id'], result['formatted_time'])
        bm25_score = bm25_weight * (1.0 - (i / len(bm25_results)))
        
        if key in all_results:
            all_results[key]['score'] += bm25_score
            all_results[key]['sources'].append('bm25')
        else:
            all_results[key] = {
                'data': result,
                'score': bm25_score,
                'sources': ['bm25']
            }
    
    # Sort by combined score
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: x['score'],
        reverse=True
    )[:k]
    
    # Format final results
    return [
        {
            **result['data'],
            'combined_score': result['score'],
            'sources': result['sources']
        }
        for result in sorted_results
    ]
