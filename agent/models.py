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


class VectorStore:
    def __init__(self, params: CreateVectorStore):
        if not openai.api_key:
            openai.api_key = config.OPENAI_API_KEY
        if not openai.organization:
            openai.organization = config.OPENAI_ORG_KEY

        pinecone.init(
            api_key=config.PINECONE_API_KEY, environment=config.PINECONE_ENVIRONMENT
        )
        self.namespace = params.agent_id + re.sub(
            re.compile("[^\x00-\x7F]+"), "", params.objective
        )
        if params.results_store_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=params.results_store_name,
                metric="cosine",
                dimension=1536,
                pod_type="p1",
            )
        self.index = pinecone.Index(params.results_store_name)

    def embed(self, text: str) -> list:  # type: ignore
        text = text.replace("\n", " ")
        return openai.Embedding.create(  # type: ignore
            input=[text], model="text-embedding-ada-002"
        )["data"][0]["embedding"]

    def add(self, params: VectorStoreAdd):
        vector = self.embed(params.result)
        metadata = {"task": params.task.task_name, "result": params.result}
        metadata.update(params.metadata)

        self.index.upsert(
            [(params.result_id, vector, metadata)],
            namespace=self.namespace,
        )

    def query(self, params: VectorStoreQuery) -> List[VectorStoreQueryResult]:
        query_vector = self.embed(params.query)
        query_result = self.index.query(
            vector=query_vector,
            top_k=params.top_results_num,
            include_metadata=True,
            namespace=self.namespace,
            filter=params.filter,
        )
        results = [
            VectorStoreQueryResult(
                id=item["id"],
                score=item["score"],
                metadata=item["metadata"],
                values=item["values"],
            )
            for item in query_result["matches"]
        ]
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        return sorted_results
