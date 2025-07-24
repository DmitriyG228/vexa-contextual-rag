# Vexa Contextual RAG

A hybrid search system that combines semantic search (Qdrant) and text search (Elasticsearch BM25) for enhanced retrieval augmented generation (RAG) capabilities, optimized for processing data from the [Vexa API](https://github.com/Vexa-ai/vexa).

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
- **Vexa API Optimization**: Specifically designed to process data from the Vexa platform
- **Contextual Processing**: Advanced content processing and indexing with document context preservation
- **Docker Support**: Easy deployment with Docker Compose
- **LLM Integration**: OpenAI integration for RAG applications

## Vexa API Integration

This implementation is specifically optimized to process data accessed from the [Vexa API](https://github.com/Vexa-ai/vexa). The system includes:

- **Vexa Data Format Support**: Handles Vexa's specific data structures and content types
- **Contextual Processing Pipeline**: Preserves conversation context and speaker information
- **Optimized Indexing**: Efficient processing of Vexa's conversational data
- **Metadata Preservation**: Maintains Vexa-specific metadata like speaker information, timestamps, and conversation flow

## Architecture

The system consists of several key components:

- **Contextual Embedding Engine**: Implements Anthropic's methodology for context-aware document processing
- **Search Engine**: Hybrid search combining Qdrant and Elasticsearch with contextual awareness
- **Indexing Pipeline**: Content processing and vector generation with document context preservation
- **LLM Interface**: OpenAI integration with streaming support
- **Vexa Data Processor**: Specialized handling for Vexa API data formats
- **Storage**: Persistent storage for vectors and documents with contextual metadata

## Contextual Retrieval Process

1. **Document Analysis**: Analyze the full document to understand its structure and context
2. **Context Extraction**: Extract relevant contextual information for each chunk
3. **Contextual Preprocessing**: Prepend context to each chunk before embedding
4. **Hybrid Indexing**: Store both semantic vectors and text indices with contextual information
5. **Intelligent Retrieval**: Use contextual awareness to improve search relevance

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- OpenAI API key
- Access to Vexa API (optional, for Vexa data processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vexa-contextual-rag
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and Vexa API credentials
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

#### Contextual Hybrid Search

```python
from search.hybrid_search import search
from search.qdrant import QdrantSearchEngine
from search.bm25 import ElasticsearchBM25

# Initialize search engines
qdrant_engine = QdrantSearchEngine()
es_engine = ElasticsearchBM25()

# Perform contextual hybrid search
results = await search(
    query="your search query",
    qdrant_engine=qdrant_engine,
    es_engine=es_engine,
    k=10
)

print(f"Found {results['total']} results")
for result in results['results']:
    print(f"Score: {result['score']}, Content: {result['content'][:100]}...")
    print(f"Context: {result['contextualized_content'][:100]}...")
```

#### Contextual Content Indexing

```python
from indexing.processor import ContentProcessor

processor = ContentProcessor()

# Process and index content with context preservation
await processor.process_content(
    content="Your content to index",
    content_id="unique_id",
    content_type="text",
    document_context="Full document context for better retrieval"
)
```

#### Vexa Data Processing

```python
from indexing.processor import ContentProcessor

processor = ContentProcessor()

# Process Vexa API data with contextual awareness
await processor.process_vexa_content(
    vexa_data=vexa_api_response,
    conversation_id="conv_123",
    preserve_speaker_context=True
)
```

#### LLM Integration with Context

```python
from llm import generic_call, user_msg, system_msg

messages = [
    system_msg("You are a helpful assistant with access to contextual information."),
    user_msg("Answer based on the provided contextualized content: [contextualized_content]")
]

response = await generic_call(messages, model="gpt-4o-mini")
print(response)
```

## Project Structure

```
vexa-contextual-rag/
├── search/                 # Search engine modules
│   ├── hybrid_search.py   # Main hybrid search implementation
│   ├── qdrant.py         # Qdrant vector search
│   ├── bm25.py           # Elasticsearch BM25 search
│   └── __init__.py
├── indexing/              # Content processing
│   ├── processor.py      # Main content processor with contextual awareness
│   ├── models.py         # Data models
│   ├── prompts.py        # LLM prompts for contextual processing
│   └── instructor_models.py
├── llm.py                # LLM interface
├── docker-compose.yaml   # Service orchestration
├── Dockerfile           # Application container
└── README.md           # This file
```

## API Reference

### Contextual Hybrid Search

The main search function combines results from both semantic and text search engines with contextual awareness:

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
- `content_ids`: Optional list of content IDs to filter by
- `k`: Maximum number of results to return

**Returns:**
- Dictionary with `results` list and `total` count
- Each result includes both original content and contextualized content

### Contextual Content Processing

```python
async def process_content(
    content: str,
    content_id: str,
    content_type: str = "text",
    metadata: Optional[Dict] = None,
    document_context: Optional[str] = None
)
```

### Vexa-Specific Processing

```python
async def process_vexa_content(
    vexa_data: Dict,
    conversation_id: str,
    preserve_speaker_context: bool = True
)
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
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
- **Reduced Latency**: Efficient processing with contextual awareness
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