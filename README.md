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


## Vexa API Integration

This implementation is specifically optimized to process meeting data accessed from the [Vexa API](https://github.com/Vexa-ai/vexa). The system includes:

- **Meeting Transcript Processing**: Handles Vexa's meeting transcript format with segments, speakers, and timestamps
- **Speaker-Aware Chunking**: Groups consecutive messages by speaker and topic for better context preservation
- **Contextual Processing Pipeline**: Preserves conversation context and speaker information
- **Optimized Indexing**: Efficient processing of meeting data with metadata preservation
- **Filtering**: Enables filtering by specific speakers and meeting IDs for targeted retrieval


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
   git clone https://github.com/Vexa-ai/vexa-contextual-rag
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

#### see usage.ipynb


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT

## Support

For issues and questions, please open an issue on the GitHub repository.

## References

- [Anthropic Contextual Embeddings Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)
- [Milvus Contextual Retrieval Tutorial](https://milvus.io/docs/contextual_retrieval_with_milvus.md)
- [Vexa API Documentation](https://github.com/Vexa-ai/vexa) 
