import threading

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from htn_planner import HTNPlanner
from guidance import models

from llm_utils import get_initial_task, compress_capabilities
from text_utils import trace_function_calls

import openai

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@trace_function_calls
def task_node_to_dict(task_node):
    if task_node is None:
        return None

    return {
        "task_name": task_node.task_name,
        "status": task_node.status,
        "children": [task_node_to_dict(child) for child in task_node.children]
    }

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def send_task_node_update(task_node):
    root_task_node = task_node
    while root_task_node.parent is not None:
        root_task_node = root_task_node.parent
    task_node_data = task_node_to_dict(root_task_node)
    socketio.emit('task_node_update', task_node_data)

def run_server():
    socketio.run(app, host="127.0.0.1", debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True, log_output=False)

def print_plan(task_node, depth=0):
    print(f"{'  ' * depth}- {task_node.task_name}")
    for child in task_node.children:
        print_plan(child, depth + 1)

def main(fast_run=False):
    with open('function_trace.log', 'w') as log_file:
        log_file.write("")

    if fast_run:
        initial_state_input = "astro and playmate are on a beach"
        goal_input = "astro and playmate explore the beach to try and find a treasure map"
    else:
        initial_state_input = input("Describe the initial state: ")
        goal_input = input("Describe your goal: ")

    # Set default capabilities to a Linux terminal with internet access
    default_capabilities = "Linux terminal, internet access"
    print(f"Default capabilities: {default_capabilities}")

    if fast_run:
        capabilities_input = ""
    else:
        capabilities_input = input("Describe the capabilities available (press Enter to use default): ")

    if not capabilities_input.strip():
        capabilities_input = default_capabilities

    lm=models.TransformersChat("NousResearch/Nous-Hermes-2-Mistral-7B-DPO")

    goal_task = get_initial_task(lm, goal_input)
    compressed_capabilities = compress_capabilities(lm, capabilities_input)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    print("Starting planning with the initial goal task:", goal_task)

    htn_planner = HTNPlanner(initial_state_input, goal_task, compressed_capabilities, 5, send_task_node_update, lm)
    plan = htn_planner.htn_planning()

    if plan:
        print("Plan found:")
        print_plan(plan)
    else:
        print("No plan found.")

if __name__ == '__main__':
    main(fast_run=False)