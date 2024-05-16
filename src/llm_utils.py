import datetime
import os
from text_utils import trace_function_calls
from guidance_prompts import htn_prompts
from guidance import models
from api import log_response

# Determines if the current world state matches the goal state
@trace_function_calls
def is_goal(state, lm, goal_task):
    response = lm + (f"Given the current state '{state}' and the goal '{goal_task}', "
              f"determine if the current state satisfies the goal. "
              f"Please provide the answer as 'True' or 'False':")

    log_response("is_goal", response)
    return response == "true"


# Provides an initial high level task that is likely to meet the goal requirements to start performing decomposition from
@trace_function_calls
def get_initial_task(lm, goal):
    response = lm + f"Given the goal '{goal}', suggest a high level task that will complete it:"
    log_response("get_initial_task", response)
    return response


@trace_function_calls
def is_task_primitive(task_name, capabilities_text):
    response = htn_prompts.is_task_primitive(task_name, capabilities_text)
    return response == "primitive"


@trace_function_calls
def compress_capabilities(lm, text):
    response = lm + f"Compress the capabilities description '{text}' into a more concise form:"
    return response


# Needs pre-conditions to prevent discontinuities in the graph
@trace_function_calls
def can_execute(lm, task, capabilities, state):
    prompt = (f"Given the task '{task}', the capabilities '{capabilities}', "
              f"and the state '{state}', determine if the task can be executed. "
              f"Please provide the answer as 'True' or 'False':")

    response = lm.gen(prompt)

    log_response("can_execute", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip().lower() == "true"


def log_state_change(prev_state, new_state, task):
    log_dir = "../state_changes"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/state_changes.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Executing task '{task}'\n")
        log_file.write(f"{timestamp}: Previous state: '{prev_state}'\n")
        log_file.write(f"{timestamp}: New state: '{new_state}'\n\n")
