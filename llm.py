import instructor
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import Literal, List, Optional, AsyncGenerator, Tuple, Any, Dict, ClassVar

import re
import os

from dotenv import load_dotenv
import tiktoken
from pydantic import BaseModel



# Load environment variables from .env file
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if not all([OPENAI_API_KEY]):
    raise ValueError("One or more required API keys are missing from the .env file")

from litellm import acompletion

@dataclass
class Msg:
    role: Literal['user','system','assistant']
    content: str
    stage: Optional[str] = None
    service_content: Optional[dict] = None

def assistant_msg(msg, service_content=None):
    return Msg(role='assistant', content=msg, service_content=service_content)
def user_msg(msg):
    return Msg(role='user', content=msg)
def system_msg(msg):
    return Msg(role='system', content=msg)

async def generic_call_(messages: List[Msg], model='default', temperature=0, max_tokens=4000, timeout=60, streaming=False) -> AsyncGenerator[str, None]:
    if model == 'default': model = "gpt-4o-mini"
    if model == 'turbo': model = "gpt-4o"
    
    messages_dict = [msg.__dict__ for msg in messages]
    for msg in messages_dict:
        if 'service_content' in msg and msg['service_content'] is not None:
            # If service_content is a dict with 'output', use that
            if isinstance(msg['service_content'], dict) and 'output' in msg['service_content']:
                msg['content'] = msg['service_content']['output']
            del msg['service_content']
    messages_dict = [{k: v for k, v in d.items() if k != 'stage'} for d in messages_dict]


    try:
        results = await acompletion(
            temperature=temperature,
            model=model,
            messages=messages_dict,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=streaming
        )
       
    except Exception as e:
        print(f"Error with provided model {model}. Falling back to default model.")
        model = "gpt-4o-mini"
        results = await acompletion(
            temperature=temperature,
            model=model,
            messages=messages_dict,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=streaming
        )
    
    if streaming:
        async for chunk in results:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    else:
        yield results.choices[0].message.content
        
async def generic_call(messages: List[Msg], model='default', temperature=0, max_tokens=4000, timeout=60):
    result = [token async for token in generic_call_(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, streaming=False)]
    print(''.join(result))
    return ''.join(result)


async def generic_call_stream(messages: List[Msg], model='default', temperature=0, max_tokens=4000, timeout=60, use_cache=False):
    output = ""
    async for token in generic_call_(messages, model=model, temperature=temperature, max_tokens=max_tokens, timeout=timeout, streaming=True, use_cache=use_cache):
        print(token, end='', flush=True)
        output += token
    print()  # New line after completion
    return output

class BaseCall(BaseModel):
    @classmethod
    async def call(cls, messages: List[Msg], model='default', temperature=0.) -> Any:
        """Call the model without streaming support"""
        if model == 'default':
            model = "gpt-4o-mini"
        if model == 'turbo':
            model = "gpt-4o"

        # Fix message format for new OpenAI API
        messages_dict = []
        for msg in messages:
            # Convert content to string if it's not already
            content = str(msg.content) if isinstance(msg.content, (list, dict)) else msg.content
            messages_dict.append({
                'role': msg.role,
                'content': content
            })
     

        try:
            client = instructor.patch(AsyncOpenAI())
            
            # Regular non-streaming response
            response = await client.chat.completions.create(
                model=model,
                messages=messages_dict,
                temperature=temperature,
                response_model=cls,
                stream=False
            )
          
            return response

        except Exception as e:
            print(f'Error in BaseCall.call_sync: {str(e)}')
            return None
    
    def get(self):
        """Get model as JSON string"""
        return self.model_dump_json(indent=2)
    
    def print(self):
        """Print model as formatted JSON"""
        print(self.get())

    @classmethod
    async def call_stream(cls, messages: List[Msg], model='default', temperature=0.) -> AsyncGenerator[Any, None]:
        """Stream the model response with partial results"""
        if model == 'default':
            model = "gpt-4o-mini"
        if model == 'turbo':
            model = "gpt-4o"

        # Fix message format for new OpenAI API
        messages_dict = []
        for msg in messages:
            # Convert content to string if it's not already
            content = str(msg.content) if isinstance(msg.content, (list, dict)) else msg.content
            messages_dict.append({
                'role': msg.role,
                'content': content
            })

        client = instructor.patch(AsyncOpenAI())
        
        # Create streaming response using Partial type
        stream = await client.chat.completions.create(
            model=model,
            messages=messages_dict,
            temperature=temperature,
            response_model=instructor.Partial[cls],
            stream=True
        )
        
        async for partial_response in stream:
            yield partial_response


class MostRelevantToChosen(BaseCall):
    title: Optional[str] = None

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a given string.
    
    Args:
    text (str): The input string to tokenize.
    model (str): The name of the model to use for tokenization (default: "gpt-4o-mini").
    
    Returns:
    int: The number of tokens in the input string.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Model {model} not found. Falling back to cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
    
def clean_text(text: str) -> str:
    return re.sub(' +', ' ', text)
