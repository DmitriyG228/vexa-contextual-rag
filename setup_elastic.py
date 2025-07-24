from search.bm25 import ElasticsearchBM25
import asyncio

async def setup_elastic():
    # Create and initialize search engine
    search_engine = await ElasticsearchBM25.create()
    if not search_engine.es_client:
        print("Failed to initialize Elasticsearch client")
        return
    # Drop existing index if exists
    exists = await search_engine.es_client.indices.exists(index=search_engine.index_name)
    if exists:
        print(f"Dropping index: {search_engine.index_name}")
        await search_engine.es_client.indices.delete(index=search_engine.index_name)
    # Create new index
    await search_engine.create_index()
    print("Elasticsearch index setup completed successfully")
    # Properly close the client
    await search_engine.es_client.close()

if __name__ == "__main__":
    asyncio.run(setup_elastic())