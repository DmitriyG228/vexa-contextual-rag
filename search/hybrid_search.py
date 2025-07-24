from typing import List, Dict, Any, Optional
from search.qdrant import QdrantSearchEngine
from search.bm25 import ElasticsearchBM25

async def search(
    query: str,
    qdrant_engine: Optional[QdrantSearchEngine] = None,
    es_engine: Optional[ElasticsearchBM25] = None,
    content_ids: Optional[List[str]] = None,
    k: int = 100
) -> Dict:
    # Get semantic search results
    semantic_results = []
    if qdrant_engine:
        semantic_results = await qdrant_engine.search(
            query=query,
            content_ids=content_ids,
            k=k
        )
    
    # Get text search results
    text_results = []
    if es_engine:
        text_results = await es_engine.search(
            query=query,
            content_ids=content_ids,
            k=k
        )
        
    # Merge results
    all_results = []
    seen_content = set()
    
    # Process semantic results first
    for result in semantic_results:
        content_id = result.payload.get('content_id')
        if content_id not in seen_content:
            seen_content.add(content_id)
            all_results.append({
                'score': result.score,
                'content': result.payload.get('content', ''),
                'content_id': content_id,
                'timestamp': result.payload.get('timestamp'),
                'formatted_time': result.payload.get('formatted_time'),
                'contextualized_content': result.payload.get('contextualized_content', ''),
                'content_type': result.payload.get('content_type', ''),
                'topic': result.payload.get('topic', ''),
                'speaker': result.payload.get('speaker', ''),
                'speakers': result.payload.get('speakers', [])
            })
    
    # Process text results
    if text_results:
        for hit in text_results['hits']['hits']:
            content_id = hit['_source'].get('content_id')
            if content_id not in seen_content:
                seen_content.add(content_id)
                all_results.append({
                    'score': hit['_score'],
                    'content': hit['_source'].get('content', ''),
                    'content_id': content_id,
                    'timestamp': hit['_source'].get('timestamp'),
                    'formatted_time': hit['_source'].get('formatted_time'),
                    'contextualized_content': hit['_source'].get('contextualized_content', ''),
                    'content_type': hit['_source'].get('content_type', ''),
                    'topic': hit['_source'].get('topic', ''),
                    'speaker': hit['_source'].get('speaker', ''),
                    'speakers': hit['_source'].get('speakers', [])
                })
    
    # Sort by score descending
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    return {
        'results': all_results[:k],
        'total': len(all_results)
    }

# Example usage in a notebook: 