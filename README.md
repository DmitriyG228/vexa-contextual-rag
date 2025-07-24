# Vexa Contextual RAG

A hybrid search system that combines semantic search (Qdrant) and text search (Elasticsearch BM25) for enhanced retrieval augmented generation (RAG) capabilities, optimized for processing meeting data from the [Vexa API](https://github.com/Vexa-ai/vexa).

## What are Contextual Embeddings?

Contextual embeddings, pioneered by [Anthropic](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings), address a critical limitation in traditional RAG systems: **semantic isolation of chunks**. 

### The Problem with Traditional RAG

In conventional RAG implementations, documents are divided into chunks for vector storage and retrieval. However, this chunking process often results in the loss of contextual information, making it difficult for the retriever to determine relevance accurately. For example, a chunk containing "the function returns an error" loses crucial context about which function, what type of error, and under what conditions.

### How Contextual Embeddings Solve This

Contextual embeddings enhance document chunks by prepending relevant context from the entire document before embedding or indexing. This approach:

1. **Preserves Semantic Context**: Each chunk retains information about its broader document context
2. **Improves Retrieval Accuracy**: Better matching between queries and relevant content
3. **Reduces Retrieval Errors**: More precise document selection for RAG applications
4. **Cost-Effective Processing**: When combined with prompt caching, contextualized chunks cost approximately $1.02 per million document tokens

### Key Benefits

- **Document Enhancement**: Preprocessing documents with LLMs before indexing brings documents closer to queries in terms of relevance
- **Low-Cost Processing**: Leverages KVCache for reuse of intermediate results when long context remains the same
- **Scalable Architecture**: Efficient handling of large knowledge bases with improved performance

## Features

- **Contextual Embeddings**: Implements Anthropic's contextual retrieval methodology
- **Hybrid Search**: Combines semantic and text-based search for better results
- **Qdrant Integration**: Vector database for semantic similarity search
- **Elasticsearch BM25**: Traditional text search with ranking
- **Meeting Data Processing**: Specifically designed to process meeting transcripts with speaker information
- **Speaker and Meeting Filtering**: Downstream filtering capabilities for specific speakers and meeting IDs
- **Contextual Processing**: Advanced content processing and indexing with document context preservation
- **Docker Support**: Easy deployment with Docker Compose
- **LLM Integration**: OpenAI integration for RAG applications

## Vexa API Integration

This implementation is specifically optimized to process meeting data accessed from the [Vexa API](https://github.com/Vexa-ai/vexa). The system includes:

- **Meeting Transcript Processing**: Handles Vexa's meeting transcript format with segments, speakers, and timestamps
- **Speaker-Aware Chunking**: Groups consecutive messages by speaker and topic for better context preservation
- **Contextual Processing Pipeline**: Preserves conversation context and speaker information
- **Optimized Indexing**: Efficient processing of meeting data with metadata preservation
- **Downstream Filtering**: Enables filtering by specific speakers and meeting IDs for targeted retrieval

## Architecture

The system consists of several key components:

- **Contextual Embedding Engine**: Implements Anthropic's methodology for context-aware document processing
- **Search Engine**: Hybrid search combining Qdrant and Elasticsearch with contextual awareness
- **Indexing Pipeline**: Content processing and vector generation with document context preservation
- **Meeting Data Processor**: Specialized handling for Vexa meeting transcript formats
- **LLM Interface**: OpenAI integration with streaming support
- **Storage**: Persistent storage for vectors and documents with contextual metadata

## Contextual Retrieval Process

1. **Meeting Data Analysis**: Parse meeting transcripts with speaker and timestamp information
2. **Speaker-Topic Chunking**: Group consecutive messages by speaker and extract topics using LLM
3. **Context Extraction**: Extract relevant contextual information for each chunk
4. **Contextual Preprocessing**: Prepend context to each chunk before embedding
5. **Hybrid Indexing**: Store both semantic vectors and text indices with contextual information
6. **Intelligent Retrieval**: Use contextual awareness to improve search relevance
7. **Downstream Filtering**: Enable filtering by speakers and meeting IDs

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- OpenAI API key
- Voyage API key (for embeddings)
- Access to Vexa API (optional, for meeting data processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vexa-contextual-rag
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start the services**
   ```bash
   docker-compose up -d
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Meeting Data Processing

```python
import pandas as pd
import json
from indexing.processor import IndexingProcessor
from search.qdrant import QdrantSearchEngine
from search.bm25 import ElasticsearchBM25

# Initialize search engines
qdrant_engine = QdrantSearchEngine()
es_engine = ElasticsearchBM25()
await es_engine.initialize()

# Initialize processor
processor = IndexingProcessor(qdrant_engine=qdrant_engine, es_engine=es_engine)

# Load meeting data (from Vexa API or local file)
with open('data/meeting.json', 'r') as f:
    meeting_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(meeting_data['segments'])

# Extract metadata
speakers = df['speaker'].dropna().unique().tolist()
start_datetime = pd.to_datetime(df['absolute_start_time'].min())
content_id = meeting_data['id']  # Use meeting ID as content_id

# Process and index meeting data
r = await processor._merge_chunks(df, content_id, start_datetime, speakers)
await processor._index_to_search_engines(r[0], r[1])
```

#### Contextual Hybrid Search with Filtering

```python
from search.hybrid_search import search

# Basic search
results = await search(
    query="challenge",
    qdrant_engine=qdrant_engine,
    es_engine=es_engine,
    k=10
)

# Search with content_id filtering (specific meeting)
results = await search(
    query="hackathon",
    qdrant_engine=qdrant_engine,
    es_engine=es_engine,
    content_ids=["3863"],  # Filter by meeting ID
    k=10
)

# Process results with speaker information
for result in results['results']:
    print(f"Speaker: {result['speaker']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Context: {result['contextualized_content'][:100]}...")
    print(f"Topic: {result['topic']}")
    print(f"Meeting ID: {result['content_id']}")
    print("---")
```

#### Downstream Filtering

The system enables powerful downstream filtering capabilities:

```python
# Filter results by specific speakers
speaker_results = [r for r in results['results'] if r['speaker'] == 'Garbhit Sharma']

# Filter by multiple speakers
target_speakers = ['Garbhit Sharma', 'Dmitry Grankin']
speaker_filtered = [r for r in results['results'] if r['speaker'] in target_speakers]

# Filter by meeting IDs
meeting_results = [r for r in results['results'] if r['content_id'] == '3863']

# Filter by topics
topic_results = [r for r in results['results'] if 'hackathon' in r['topic'].lower()]
```

#### Advanced Search with Speaker Filtering

```python
# Use the hybrid_search function for more advanced filtering
from search.bm25 import hybrid_search

results = await hybrid_search(
    query="cybersecurity",
    qdrant_engine=qdrant_engine,
    es_engine=es_engine,
    meeting_ids=["3863"],  # Filter by meeting IDs
    speakers=["Garbhit Sharma"],  # Filter by speakers
    k=10,
    semantic_weight=0.7,
    bm25_weight=0.3
)
```

## Project Structure

```
vexa-contextual-rag/
├── search/                 # Search engine modules
│   ├── hybrid_search.py   # Main hybrid search implementation
│   ├── qdrant.py         # Qdrant vector search with content_id filtering
│   ├── bm25.py           # Elasticsearch BM25 search with filtering
│   └── __init__.py
├── indexing/              # Content processing
│   ├── processor.py      # Meeting data processor with speaker-topic chunking
│   ├── models.py         # Data models for search documents
│   ├── prompts.py        # LLM prompts for contextual processing
│   └── instructor_models.py
├── llm.py                # LLM interface
├── usage.ipynb           # Example usage notebook
├── docker-compose.yaml   # Service orchestration
├── Dockerfile           # Application container
└── README.md           # This file
```

## API Reference

### Meeting Data Processing

```python
async def _merge_chunks(
    df: pd.DataFrame, 
    content_id: str, 
    start_datetime: datetime, 
    speakers: List[str]
) -> Tuple[List[Dict], List[PointStruct]]
```

**Parameters:**
- `df`: DataFrame with meeting segments (speaker, text, timestamps)
- `content_id`: Meeting ID for filtering
- `start_datetime`: Meeting start time
- `speakers`: List of speakers in the meeting

### Contextual Hybrid Search

The main search function with content_id filtering:

```python
async def search(
    query: str,
    qdrant_engine: Optional[QdrantSearchEngine] = None,
    es_engine: Optional[ElasticsearchBM25] = None,
    content_ids: Optional[List[str]] = None,
    k: int = 100
) -> Dict
```

**Parameters:**
- `query`: Search query string
- `qdrant_engine`: Qdrant search engine instance
- `es_engine`: Elasticsearch BM25 engine instance
- `content_ids`: Optional list of meeting IDs to filter by
- `k`: Maximum number of results to return

**Returns:**
- Dictionary with `results` list and `total` count
- Each result includes speaker, topic, meeting ID, and contextualized content

### Advanced Hybrid Search

```python
async def hybrid_search(
    query: str,
    qdrant_engine,
    es_engine: ElasticsearchBM25,
    meeting_ids: List[str] = None,
    speakers: List[str] = None,
    k: int = 10,
    semantic_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> List[Dict]
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here
VEXA_API_KEY=your_vexa_api_key_here
VEXA_API_URL=https://api.vexa.ai
QDRANT_HOST=localhost
QDRANT_PORT=6333
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
```

### Docker Services

The system runs the following services:

- **Qdrant**: Vector database (port 6333)
- **Elasticsearch**: Text search engine (port 9200)

## Performance Benefits

Based on [Anthropic's research](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings) and [Milvus contextual retrieval benchmarks](https://milvus.io/docs/contextual_retrieval_with_milvus.md), this implementation provides:

- **Improved Pass@5**: Up to 90.91% accuracy with contextual retrieval + reranking
- **Enhanced Relevance**: Better matching between queries and relevant content
- **Speaker-Aware Processing**: Intelligent chunking by speaker and topic
- **Efficient Filtering**: Fast filtering by meeting IDs and speakers
- **Cost Optimization**: Leverages caching for repeated context processing

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Use a linter like `flake8` or `black` for code formatting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions, please open an issue on the GitHub repository.

## References

- [Anthropic Contextual Embeddings Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)
- [Milvus Contextual Retrieval Tutorial](https://milvus.io/docs/contextual_retrieval_with_milvus.md)
- [Vexa API Documentation](https://github.com/Vexa-ai/vexa) 