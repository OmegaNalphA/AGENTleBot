import re
import pinecone
import openai
from typing import Deque, Dict, List, Optional
from collections import deque
from pydantic import BaseModel
from config import Config

config = Config()


# Pydantic models
class Task(BaseModel):
    task_id: str
    task_name: str


class CreateAgentParams(BaseModel):
    objective: str
    log: bool = False


class CreateVectorStore(BaseModel):
    objective: str
    results_store_name: str
    agent_id: str


class VectorStoreAdd(BaseModel):
    task: Task
    result: str
    result_id: str
    metadata: Dict[str, str] = {}


class VectorStoreQuery(BaseModel):
    query: str
    top_results_num: int = 5
    filter: Optional[Dict[str, str | List[str]]] = None


class ExpectedAgentMetadata(BaseModel):
    task: str
    result: str
    thought_type: str


class VectorStoreQueryResult(BaseModel):
    id: str
    score: float
    metadata: ExpectedAgentMetadata
    values: List[float]


# Util classes
class SingleTaskListStorage:
    def __init__(self):
        self.tasks: Deque[Task] = deque([])
        self.task_id_counter = 0

    def append(self, task: Task):
        self.tasks.append(task)

    def replace(self, tasks: List[Task]):
        self.tasks = deque(tasks)

    def popleft(self):
        return self.tasks.popleft()

    def is_empty(self):
        return False if self.tasks else True

    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    def get_task_names(self):
        return [t.task_name for t in self.tasks]
