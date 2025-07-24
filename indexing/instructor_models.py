
from llm import BaseCall 
from pydantic import Field

class TopicsMapping(BaseCall):
    absolute_start_time: str = Field(..., description="time of the input qoute")
    topic: str = Field(..., description="topic of the input qoute")
    
class TopicsExtraction(BaseCall):
    mapping: list[TopicsMapping] = Field(..., description="Mapping of topics to the input text. Aim for equal chunks of time for each topic switch. ")