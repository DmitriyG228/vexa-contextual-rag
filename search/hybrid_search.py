from typing import List, Dict, Any, Optional
from search.qdrant import QdrantSearchEngine
from search.bm25 import ElasticsearchBM25
import pandas as pd

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
        
    # Convert results to pandas DataFrames for vectorized operations
    semantic_df = pd.DataFrame([
        {
            'score': result.score,
            'content': result.payload.get('content', ''),
            'content_id': result.payload.get('content_id'),
            'chunk_index': result.payload.get('chunk_index'),
            'timestamp': result.payload.get('timestamp'),
            'formatted_time': result.payload.get('formatted_time'),
            'contextualized_content': result.payload.get('contextualized_content', ''),
            'content_type': result.payload.get('content_type', ''),
            'topic': result.payload.get('topic', ''),
            'speaker': result.payload.get('speaker', ''),
            'speakers': result.payload.get('speakers', [])
        }
        for result in semantic_results
    ])
    
    semantic_df['source'] = 'qdrant'
    
    es_df = pd.DataFrame([
        {
            'score': hit['_score'],
            'content': hit['_source'].get('content', ''),
            'content_id': hit['_source'].get('content_id'),
            'chunk_index': hit['_source'].get('chunk_index'),
            'timestamp': hit['_source'].get('timestamp'),
            'formatted_time': hit['_source'].get('formatted_time'),
            'contextualized_content': hit['_source'].get('contextualized_content', ''),
            'content_type': hit['_source'].get('content_type', ''),
            'topic': hit['_source'].get('topic', ''),
            'speaker': hit['_source'].get('speaker', ''),
            'speakers': hit['_source'].get('speakers', [])
        }
        for hit in text_results.get('hits', {}).get('hits', [])
    ])
    
    es_df['source'] = 'es'
    
  
    
    # Concatenate DataFrames
    all_results_df = pd.concat([semantic_df, es_df], ignore_index=True)
    
    if all_results_df.empty:
        return {'results': [], 'total': 0}
    
    # Sort by score descending
    all_results_df = all_results_df.sort_values('score', ascending=False)
    
    # Vectorized deduplication based on content_id and chunk_index combination
    all_results_df['chunk_key'] = all_results_df['content_id'].astype(str) + '_' + all_results_df['chunk_index'].astype(str)
    deduplicated_df = all_results_df.drop_duplicates(subset=['chunk_key'], keep='first')
    
    # Remove the temporary chunk_key column
    deduplicated_df = deduplicated_df.drop('chunk_key', axis=1)
    
    # Convert back to list of dictionaries
    results = deduplicated_df.to_dict('records')
    
    return {
        'results': results,
        'total': len(deduplicated_df)
    }
