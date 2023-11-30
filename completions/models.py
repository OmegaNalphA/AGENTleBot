from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel


class PromptMessageRoleEnum(str, Enum):
    system = "system"
    user = "user"


class PromptMessage(BaseModel):
    role: Union[PromptMessageRoleEnum, str] = PromptMessageRoleEnum.system
    content: str


class ModelEnum(str, Enum):
    gpt3 = "gpt-3"
    gpt4 = "gpt-4"
    gpt_3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt_4_1106_preview = "gpt-4-1106-preview"


class ChatCompletionCreateParams(BaseModel):
    messages: List[PromptMessage] = []
    model: str = ModelEnum.gpt_3_5_turbo_16k.value
    temperature: float = 0.5
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = 0.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[str | List[str]] = None
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, Any]] = None


# Response Pydantic Models
class ChatCompletionResponseChoiceMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    finish_reason: str
    index: int
    message: ChatCompletionResponseChoiceMessage


class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[ChatCompletionResponseChoice]
    created: int
    model: str
    object: str
    usage: Dict[Any, Any]
