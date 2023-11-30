from typing import List

from agent.models import ExpectedAgentMetadata


def stringify_context(metadata: ExpectedAgentMetadata) -> str:
    match metadata.thought_type:
        case "INTERNAL_THOUGHT":
            return f"""Based on the task "{metadata.task}", 
            you had the private thought "{metadata.result}" """
        case "EXECUTE_THOUGHT":
            return f"""Based on the task "{metadata.task}", 
            you executed the response "{metadata.result}" """
        case _:
            return f"""Based on the task "{metadata.task}", 
            you had the thought "{metadata.result}" """


def initial_task(objective: str) -> str:
    return f"""
    You are an autonomous agent.
    Your goal is to accomplish the following objective: {objective}
    Your first task is to develop a task list for the given objective.
    Return one task per line in your response. The result must be a 
    numbered list in the format:
    #. First task
    #. Second task
    The number of each entry must be followed by a period. 
    If your list is empty, write "There are no tasks to add at this time."
    Unless your list is empty, do not include any headers before your 
    numbered list or follow your numbered list with any other output.
    """


def internal_thought(
    objective: str, task: str, context: List[ExpectedAgentMetadata]
) -> str:
    description = f"""
    You have been given the following objective: {objective}. 
    Related to that objective, you have been given the following task: 
    {task}.
    You must think about it and plan what action to take.
    """

    thoughts = (
        f"""
    For some context, here are your memories related to the query.
    MEMORIES sorted in relevance:
    {[stringify_context(c) for c in context]}
    """
        if len(context) > 0
        else ""
    )

    call_to_action = f"""
    Think of some actions you would take after hearing about the 
    task "{task}" based on your past thoughts and actions.
    This is not shown to the outside world, but only to yourself. 
    It is just your internal thought.
    """

    return description + thoughts + call_to_action


def execute_thought(
    objective: str,
    task: str,
    internal_thought: str,
    context: List[ExpectedAgentMetadata],
) -> str:
    description = f"""
    Perform one task based on the following objective: {objective}
    Your current task is: {task}
    Based on the task, you have thought about the input and had 
    the following thought: {internal_thought}
    """

    thoughts = (
        f"""
    For some context, here are your memories related to the query.
    MEMORIES sorted in relevance:
    {[stringify_context(c) for c in context]}
    """
        if len(context) > 0
        else ""
    )

    call_to_action = """
    Return your response to the task.
    """

    return description + thoughts + call_to_action


def task_creation(
    objective: str,
    previous_execute_thought: str,
    previous_task: str,
    task_list: List[str],
):
    description = f"""
    You are to use the result from an execution agent to create new 
    tasks with the following objective: {objective}.
    The last completed task has the result: 
    {previous_execute_thought}.
    The last completed task was: {previous_task}.
    """

    thoughts = (
        f"""For some context, here are your current incomplete tasks:
        {"/n".join(task_list)}"""
        if len(task_list) > 0
        else ""
    )

    call_to_action = """Return one task per line in your response. 
    The result must be a numbered list in the format:
    
    #. First task
    #. Second task
    
    The number of each entry must be followed by a period. 
    If your list is empty, write "There are no tasks to add at this time."
    Unless your list is empty, do not include any headers 
    before your numbered list or follow your numbered list with any other output.
    """

    return description + thoughts + call_to_action


def task_prioritization(objective: str, task_list: List[str]):
    description = f"""
    You are tasked with prioritizing the following tasks: 
    {"/n".join(task_list)} 
    Consider the ultimate objective of your team: {objective}.
    """

    call_to_action = """
    Tasks should be sorted from highest to lowest priority, 
    where higher-priority tasks are those that act as pre-requisites 
    or are more essential for meeting the objective.
    Do not remove any tasks. Return the ranked tasks as a numbered 
    list in the format:
    
    #. First task
    #. Second task
    
    The entries must be consecutively numbered, starting with 1. 
    The number of each entry must be followed by a period.
    Do not include any headers before your ranked list or follow 
    your list with any other output.
    """

    return description + call_to_action
