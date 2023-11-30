from pydantic import BaseModel


class ExpectedAgentMetadata(BaseModel):
    task: str
    result: str
    thought_type: str
