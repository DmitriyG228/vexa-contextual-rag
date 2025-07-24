# Vexa Contextual RAG

A hybrid search system that combines semantic search (Qdrant) and text search (Elasticsearch BM25) for enhanced retrieval augmented generation (RAG) capabilities.

## Features

- **Hybrid Search**: Combines semantic and text-based search for better results
- **Qdrant Integration**: Vector database for semantic similarity search
- **Elasticsearch BM25**: Traditional text search with ranking
- **Contextual Processing**: Advanced content processing and indexing
- **Docker Support**: Easy deployment with Docker Compose
- **LLM Integration**: OpenAI integration for RAG applications

## Architecture

The system consists of several key components:

- **Search Engine**: Hybrid search combining Qdrant and Elasticsearch
- **Indexing Pipeline**: Content processing and vector generation
- **LLM Interface**: OpenAI integration with streaming support
- **Storage**: Persistent storage for vectors and documents

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vexa-contextual-rag
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
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

#### Basic Hybrid Search

```python
from search.hybrid_search import search
from search.qdrant import QdrantSearchEngine
from search.bm25 import ElasticsearchBM25

# Initialize search engines
qdrant_engine = QdrantSearchEngine()
es_engine = ElasticsearchBM25()

# Perform hybrid search
results = await search(
    query="your search query",
    qdrant_engine=qdrant_engine,
    es_engine=es_engine,
    k=10
)

print(f"Found {results['total']} results")
for result in results['results']:
    print(f"Score: {result['score']}, Content: {result['content'][:100]}...")
```

#### Content Indexing

```python
from indexing.processor import ContentProcessor

processor = ContentProcessor()

# Process and index content
await processor.process_content(
    content="Your content to index",
    content_id="unique_id",
    content_type="text"
)
```

#### LLM Integration

```python
from llm import generic_call, user_msg, system_msg

messages = [
    system_msg("You are a helpful assistant."),
    user_msg("Answer based on the provided context: [context]")
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
│   ├── processor.py      # Main content processor
│   ├── models.py         # Data models
│   ├── prompts.py        # LLM prompts
│   └── instructor_models.py
├── llm.py                # LLM interface
├── docker-compose.yaml   # Service orchestration
├── Dockerfile           # Application container
└── README.md           # This file
```

## API Reference

### Hybrid Search

The main search function combines results from both semantic and text search engines:

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

### Content Processing

```python
async def process_content(
    content: str,
    content_id: str,
    content_type: str = "text",
    metadata: Optional[Dict] = None
)
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
```

### Docker Services

The system runs the following services:

- **Qdrant**: Vector database (port 6333)
- **Elasticsearch**: Text search engine (port 9200)

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