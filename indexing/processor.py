from typing import List, Dict, Any, Tuple, Optional, Callable
import pandas as pd
from datetime import datetime
from qdrant_client.models import PointStruct


from .instructor_models import TopicsExtraction

from llm import generic_call, system_msg, user_msg


from .prompts import DOCUMENT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT
from .models import SearchDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import logging

logger = logging.getLogger(__name__)

class ProcessingError(Exception): pass

class IndexingProcessor:
    def __init__(self, qdrant_engine, es_engine, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.qdrant_engine = qdrant_engine
        self.es = es_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )


    async def _chunk_by_topic_and_speaker(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create chunks by grouping consecutive messages from the same speaker and topic.
        
        Args:
            df: DataFrame with columns ['absolute_start_time', 'speaker', 'text']
            
        Returns:
            DataFrame with grouped chunks containing ['absolute_start_time', 'speaker', 'topic', 'text']
        """
        # Convert to markdown for topic extraction
        input_text = df[['absolute_start_time','speaker', 'text']].to_markdown()
        
        # Use RecursiveCharacterTextSplitter with OpenAI's message length limit
        MAX_MESSAGE_LENGTH = 1000000  # Leaving some buffer
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=MAX_MESSAGE_LENGTH,
            chunk_overlap=0  # No overlap needed since we're sending all chunks in one call
        )
        chunks = text_splitter.split_text(input_text)
        
        # Create messages array with system prompt and multiple user messages
        messages = [system_msg("Extract topics from the following meeting transcript sections. Maintain consistency across sections.")]
        for chunk in chunks:
            messages.append(user_msg(chunk))
        
        # Make single API call with all chunks as separate messages
        topics_result = await TopicsExtraction.call(messages)
        
        # Create topics DataFrame and merge with original data
        topics_df = pd.DataFrame([{"absolute_start_time": m.absolute_start_time, "topic": m.topic} 
                                for m in topics_result.mapping])
        
        df = df.merge(topics_df, on='absolute_start_time', how='left')[['absolute_start_time','topic','speaker','text']].ffill()
        
        # Create speaker_shift column to identify when speaker or topic changes
        df['speaker_shift'] = (df['speaker']+df['topic'] != df['speaker']+df['topic'].shift(1)).cumsum()
        
        # Group by speaker_shift to create chunks
        df_grouped = df.groupby('speaker_shift').agg({
            'absolute_start_time': 'first',
            'speaker': 'first',
            'topic': 'first',
            'text': ' '.join
        }).reset_index()
        
        return df_grouped

    async def _merge_chunks(self, df: pd.DataFrame, content_id: str, start_datetime: datetime, speakers: List[str]) -> Tuple[List[Dict], List[PointStruct]]:
        # Use the new chunking method
        df_grouped = await self._chunk_by_topic_and_speaker(df)
        
        chunks = (df_grouped['speaker'] + ': ' + df_grouped['text']).tolist()
        doc_text = '\n'.join(chunks)
        contextualized_chunks = await self._contextualize_chunks(chunks, doc_text)
        
        return await self._prepare_search_documents(
            chunks=chunks,
            contextualized_chunks=contextualized_chunks,
            content_id=content_id,
            timestamp=start_datetime,
            topic_provider=lambda i: df_grouped.iloc[i].topic,
            speaker_provider=lambda i: df_grouped.iloc[i].speaker,
            speakers=speakers,
        )

    async def _prepare_search_documents(
        self,
        chunks: List[str],
        contextualized_chunks: List[str],
        content_id: str,
        timestamp: datetime,
        topic_provider: Callable[[int], str],
        speaker_provider: Callable[[int], str],
        speakers: List[str],
    ) -> Tuple[List[Dict], List[PointStruct]]:
        """Prepare search documents with embeddings for both Elasticsearch and Qdrant."""
        # Generate embeddings
        embeddings = self.qdrant_engine.voyage.embed(texts=contextualized_chunks, model='voyage-3').embeddings
        
        # Create search documents
        search_docs = [
            SearchDocument(
                content_id=content_id,
                timestamp=timestamp,
                chunk=chunk,
                context=context,
                chunk_index=i,
                topic=topic_provider(i),
                speaker=speaker_provider(i),
                speakers=speakers,
            )
            for i, (chunk, context) in enumerate(zip(chunks, contextualized_chunks))
        ]
        
        # Convert to ES docs and Qdrant points
        es_documents = [doc.to_es_doc() for doc in search_docs]
        qdrant_points = [doc.to_qdrant_point(emb) for doc, emb in zip(search_docs, embeddings)]
        
        return es_documents, qdrant_points

    async def _contextualize_chunks(self, chunks: List[str], doc_text: str) -> List[str]:
        messages = [
            system_msg(DOCUMENT_CONTEXT_PROMPT.format(doc_text=doc_text)),
            user_msg(CHUNK_CONTEXT_PROMPT.format(chunk_text=chunks[0]))
        ]
        first_context = await generic_call(messages)
        contextualized_chunks = [first_context]
        
        async def get_context(chunk):
            messages = [
                system_msg(DOCUMENT_CONTEXT_PROMPT.format(doc_text=doc_text)),
                user_msg(CHUNK_CONTEXT_PROMPT.format(chunk_text=chunk))
            ]
            return await generic_call(messages)
        
        remaining_contexts = await asyncio.gather(*(get_context(chunk) for chunk in chunks[1:]))
        contextualized_chunks.extend(remaining_contexts)
        return contextualized_chunks

    async def _index_to_search_engines(self, es_documents: List[Dict], qdrant_points: List[PointStruct]) -> None:
        if not es_documents or not qdrant_points:
            return
        
        for i in range(0, len(es_documents), 100):
            batch_es = es_documents[i:i+100]
            batch_qdrant = qdrant_points[i:i+100]
            
            # Use es_client instead of client
            operations = []
            for doc in batch_es:
                operations.extend([
                    {"index": {"_index": self.es.index_name}},
                    doc
                ])
            await self.es.es_client.bulk(operations=operations)
            
            await self.qdrant_engine.client.upsert(
                collection_name=self.qdrant_engine.collection_name,
                points=batch_qdrant
            )
        