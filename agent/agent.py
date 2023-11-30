import re
import time
from typing import Deque, Dict, List, Optional
from agent.models import (
    CreateAgentParams,
    CreateVectorStore,
    ExpectedAgentMetadata,
    SingleTaskListStorage,
    Task,
    VectorStore,
    VectorStoreAdd,
    VectorStoreQuery,
)
from agent.prompts import (
    execute_thought,
    initial_task,
    internal_thought,
    task_creation,
    task_prioritization,
)
from completions.models import (
    ChatCompletionCreateParams,
    PromptMessage,
    PromptMessageRoleEnum,
)
from completions.flows import llm_call as create_completion


class Agent:
    def __init__(self, params: CreateAgentParams) -> None:
        self.tasks_storage = SingleTaskListStorage()
        self.objective = params.objective
        self.tasks_storage.append(
            Task(
                task_id=str(self.tasks_storage.next_task_id()),
                task_name=initial_task(self.objective),
            )
        )
        self.agent_id = "gentle_bot"
        self.vector_store = VectorStore(
            CreateVectorStore(
                objective=self.objective,
                results_store_name="semantic-search-prototype",
                agent_id=self.agent_id,
            )
        )
        self.log = params.log

    def run(self):
        loop = True
        while loop:
            if not self.tasks_storage.is_empty():
                if self.log:
                    print(f"***Tasks left:***\n {len(self.tasks_storage.tasks)}\n---")
                # Get task from storage
                task = self.tasks_storage.popleft()
                if self.log:
                    print(f"***Current task:***\n {task.task_name}\n---")
                # Execute task
                execute_thought = self.execute_task(task)
                # Create new tasks and add to storage
                self.create_tasks(task, execute_thought)
                # Prioritize task list
                self.prioritize_tasks()
                # Sleep for a bit to not hit API limits
                time.sleep(3)
            else:
                # No tasks left, exit
                print("Done.")
                loop = False
                break

    def __get_context(
        self, query: str, filter: Optional[Dict[str, str | List[str]]] = None
    ) -> List[ExpectedAgentMetadata]:
        context = self.vector_store.query(
            VectorStoreQuery(
                query=query,
                filter=filter,
            )
        )
        return [c.metadata for c in context]

    def __internal_thought(self, task: Task):
        # Get all possible thoughts and actions to inform the internal thought
        context = self.__get_context(query=self.objective)
        internal_thought_prompt = internal_thought(
            objective=self.objective, task=task.task_name, context=context
        )
        thought: str = create_completion(
            completion_in=ChatCompletionCreateParams(
                temperature=0.8,
                messages=[
                    PromptMessage(
                        role=PromptMessageRoleEnum.user, content=internal_thought_prompt
                    )
                ],
            ),
        )
        self.vector_store.add(
            params=VectorStoreAdd(
                task=task,
                result=thought,
                result_id=f"result_{task.task_id}",
                metadata={"thought_type": "INTERNAL_THOUGHT"},
            )
        )
        if self.log:
            print(f"***Internal Thought:***\n {thought}\n---")
        return thought

    def __execute_thought(self, task: Task, internal_thought: str):
        context = self.__get_context(
            query=self.objective,
        )
        execute_thought_prompt = execute_thought(
            objective=self.objective,
            task=task.task_name,
            internal_thought=internal_thought,
            context=context,
        )
        # TODO: Make this have function calls and deal with that
        response: str = create_completion(
            completion_in=ChatCompletionCreateParams(
                temperature=0.7,
                messages=[
                    PromptMessage(
                        role=PromptMessageRoleEnum.user, content=execute_thought_prompt
                    )
                ],
            ),
        )
        self.vector_store.add(
            params=VectorStoreAdd(
                task=task,
                result=response,
                result_id=f"result_{task.task_id}",
                metadata={"thought_type": "EXECUTE_THOUGHT"},
            )
        )
        if self.log:
            print(f"***Execute Thought:***\n {response}\n---")
        return response

    def __strip_task_response(self, response: str) -> List[str]:
        new_tasks = response.split("\n") if "\n" in response else [response]
        new_tasks_list = []
        for task_string in new_tasks:
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = "".join(s for s in task_parts[0] if s.isnumeric())
                task_name = re.sub(r"[^\w\s_]+", "", task_parts[1]).strip()
                if task_name.strip() and task_id.isnumeric():
                    new_tasks_list.append(task_name)

        return new_tasks_list

    def execute_task(self, task: Task):
        internal_thought = self.__internal_thought(task)
        execute_thought = self.__execute_thought(task, internal_thought)
        return execute_thought

    def create_tasks(self, task: Task, execute_thought: str) -> Deque[Task]:
        task_creation_prompt = task_creation(
            objective=self.objective,
            previous_execute_thought=execute_thought,
            previous_task=task.task_name,
            task_list=self.tasks_storage.get_task_names(),
        )
        response: str = create_completion(
            completion_in=ChatCompletionCreateParams(
                temperature=0.6,
                messages=[
                    PromptMessage(
                        role=PromptMessageRoleEnum.user, content=task_creation_prompt
                    )
                ],
            ),
        )
        new_tasks_list = self.__strip_task_response(response)
        if self.log:
            newline_char = "\n"
            print(
                f"""***New Tasks Created:***\n 
                {newline_char.join(new_tasks_list)}\n
                ---"""
            )

        for task_name in new_tasks_list:
            self.tasks_storage.append(
                Task(
                    task_id=str(self.tasks_storage.next_task_id()), task_name=task_name
                )
            )

        return self.tasks_storage.tasks

    def prioritize_tasks(self):
        task_prioritization_prompt = task_prioritization(
            self.objective, self.tasks_storage.get_task_names()
        )
        response: str = create_completion(
            completion_in=ChatCompletionCreateParams(
                temperature=0.6,
                messages=[
                    PromptMessage(
                        role=PromptMessageRoleEnum.user,
                        content=task_prioritization_prompt,
                    )
                ],
            ),
        )
        new_tasks_list = self.__strip_task_response(response)
        if self.log:
            newline_char = "\n"
            print(
                f"""***New Tasks Prioritized:***\n
                {newline_char.join(new_tasks_list)}\n
                ---"""
            )
        self.tasks_storage.replace(
            [
                Task(
                    task_id=str(self.tasks_storage.next_task_id()), task_name=task_name
                )
                for task_name in new_tasks_list
            ]
        )

        return self.tasks_storage.tasks
