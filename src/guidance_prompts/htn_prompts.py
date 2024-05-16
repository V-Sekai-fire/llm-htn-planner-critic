import guidance
from guidance import models, gen, system, user, assistant, select
from utils import comma_seperated_items

guidance_api = None

# Add new functions for extracting and suggesting new queries here
def extract_and_format_information(lm, webpage_content):
    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Extract and format relevant information from the following webpage content: {webpage_content}
        '''

    with assistant():
        lm += gen("extracted_info")

    return lm['extracted_info']


def check_subtasks(lm, task, subtasks, capabilities_input):
    task_statuses = ['True', 'False']
    with system():
        lm += "You are a helpful agent"

    subtasks_delineated = comma_seperated_items(subtasks)

    with user():
        lm += f'''
            Given the parent task '{task}', and its subtasks '{subtasks_delineated}',
        check if these subtasks effectively and comprehensively address the requirements
        of the parent task without any gaps or redundancies, using the following capabilities:
        '{capabilities_input}'. Return 'True' if they meet the requirements or 'False' otherwise.
        '''

    with assistant():
        lm += gen("result")

    return lm["result"].strip().lower()


def get_subtasks(lm, task, state, remaining_decompositions, capabilities_input):
    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Given the task '{task}', the current state '{state}',
        and {remaining_decompositions} decompositions remaining before failing,
        please decompose the task into a detailed step-by-step plan
        that can be achieved using the following capabilities:
        '{capabilities_input}'. Provide the subtasks in a comma-separated list,
        each enclosed in square brackets: [subtask1], [subtask2], ...
        '''

    with assistant():
        lm += gen("subtasks_list")

    return lm['subtasks_list'].strip()


def suggest_new_query(query):
    new_query = guidance('''
    {{#system~}}You are a helpful assistant.{{~/system}}
    {{#user~}}Suggest a new query to find the missing information based on the initial query: {{query}}{{~/user}}
    {{#assistant}}{{gen "new_query"}}{{/assistant}}''',
                         llm=guidance_api)

    output = new_query(query=query)

    return output['new_query']


def update_plan_output(task_name, task_description, elapsed_time, time_limit, context_window):
    task_statuses = ['not started', 'in progress', 'completed']
    action_types = ['update', 'insert', 'delete']

    structured_prompt = guidance('''
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}

    {{#user~}}
    Given the task: {{task_name}}, {{task_description}} and the current state of the plan.
    What should be the task status and how should the deliverable be updated?
    Provide instructions to update, insert, or delete lines in the deliverable using quoted text as an indicator.
    The status should be one of the following: 'not started', 'in progress', 'completed'
    Elapsed time: {{elapsed_time}} seconds, time limit: {{time_limit}} seconds, context window: {{context_window}} tokens.
    {{~/user}}
    
    {{#user~}}Status:{{~/user}}
    {{#assistant~}}
    {{select "status" options=task_statuses}}
    {{~/assistant}}
    
    {{#user~}}Action:{{~/user}}
    {{#assistant~}}
    {{select "action" options=action_types}}
    {{~/assistant}}
    
    {{#user~}}Details:{{~/user}}
    {{#assistant~}}
    {{#if (eq action "update")}}{{gen "update_line"}}Update line {{update_line}} with "{{update_text}}"{{/if}}
    {{#if (eq action "insert")}}{{gen "insert_line"}}Insert "{{insert_text}}" at line {{insert_line}}{{/if}}
    {{#if (eq action "delete")}}{{gen "delete_line"}}Delete line {{delete_line}}{{/if}}
    {{~/assistant}}
    ''')

    output = structured_prompt(
        task_name=task_name,
        task_description=task_description,
        elapsed_time=elapsed_time,
        time_limit=time_limit,
        context_window=context_window,
        task_statuses=task_statuses,
        action_types=action_types
    )

    status = output['status']
    action = output['action']
    details = {}

    if action == "update":
        details["update_line"] = output["update_line"]
        details["update_text"] = output["update_text"]
    elif action == "insert":
        details["insert_line"] = output["insert_line"]
        details["insert_text"] = output["insert_text"]
    elif action == "delete":
        details["delete_line"] = output["delete_line"]

    return {"status": status, "action": action, "details": details}


def confirm_deliverable_changes(lm, deliverable_content, updated_content):

    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Please confirm the changes made to the deliverable.
        Original content:
        {deliverable_content}
        
        Updated content:
        {updated_content}
        
        Type 'yes' to confirm the changes or 'no' to revert them.
        '''

    with assistant():
        lm += gen("confirm") + select(['yes', 'no'])

    return lm['confirm']


def translate(lm, original_task, capabilities_input):
    # translates a task into a form that can be completed with the specified capabilities
    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Translate the task '{original_task}' into a form that can be executed using the following capabilities:
        '{capabilities_input}'. Provide the executable form in a single line without any commentary
        or superfluous text.
        
        When translated to use the specified capabilities the result is:
        '''

    with assistant():
        lm += gen("translated_task")

    return lm['translated_task']


@guidance
def is_task_primitive(lm, task_name, capabilities_text):
    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Given the task '{task_name}' and the capabilities '{capabilities_text}',
        determine if the task is primitive which cannot be broken up further or compound which can be broken down more.
        Please provide the answer as 'primitive' or 'compound':
        '''

    with assistant():
        lm += gen("choice")

    return lm + select(['primitive', 'compound'])


def evaluate_candidate(lm, task, subtasks, capabilities_input):
    with system():
        lm += "You are a helpful agent"

    with user():
        lm += f'''
        Given the parent task {task}, and its subtasks {subtasks}, 
        evaluate how well these subtasks address the requirements 
        of the parent task without any gaps or redundancies, using the following capabilities: 
        {capabilities_input}
        Return a score between 0 and 1, where 1 is the best possible score.
        
        Score:
        '''

    with assistant():
        lm += gen("score", temperature=0.5, max_tokens=10, regex='^[0]\.\d{{8}}$')

    return lm['score']
