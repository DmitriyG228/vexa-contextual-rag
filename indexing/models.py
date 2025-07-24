from typing import Dict, List
from datetime import datetime
from qdrant_client.models import PointStruct
import uuid
from enum import Enum

class ContentType(Enum):
    NOTE = "note"
    MEETING = "meeting"

class SearchDocumentError(Exception): pass

class SearchDocument:
    def __init__(
        self,
        content_id: str,
        timestamp: datetime,
        chunk: str,
        context: str,
        chunk_index: int,
        topic: str,
        speaker: str,
        speakers: List[str],
    ):
        # Validate inputs
        if not content_id:
            raise SearchDocumentError("content_id cannot be empty")

        self.content_id = content_id
        self.timestamp = timestamp
        self.chunk = chunk
        self.context = context
        self.chunk_index = chunk_index
        self.topic = topic
        self.speaker = speaker
        self.speakers = speakers

    @property
    def formatted_time(self) -> str:
        return self.timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def to_es_doc(self) -> Dict:
        return {
            'content_id': self.content_id,
            'timestamp': self.timestamp.isoformat(),
            'formatted_time': self.formatted_time,
            'content': self.chunk,
            'contextualized_content': self.context,
            'chunk_index': self.chunk_index,
            'topic': self.topic,
            'speaker': self.speaker,
            'speakers': self.speakers,
        }

    def to_qdrant_point(self, embedding: List[float]) -> PointStruct:
        if not embedding:
            raise SearchDocumentError("embedding cannot be empty")
            
        return PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=self.to_es_doc()
        ) 
        
