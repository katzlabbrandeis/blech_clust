"""
Check every file for top-level docstrings and update them if necessary.

Generate doc-strings using autogen LLM docstrings.

Docstrings should be in format:

General description of the file.
- Bullet points of specific functions. 
- Bullet points of specific functions. 
- Bullet points of specific functions. 
...

"""

import argparse
parser = argparse.ArgumentParser(description='Update docstrings in all python files in the specified directories')
parser.add_argument('-y', '--yes', action='store_true', help='Automatically update docstrings without prompting')
args = parser.parse_args()

import os
import sys
import json
import ast
from tqdm import tqdm
from pathlib import Path
from glob import glob
from autogen import AssistantAgent

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found in environment variables")

llm_config = {
    "model": "gpt-4o",
    "api_key": api_key,
    "temperature": 0
}

def docstring_from_file(file_path):
    """ Get the docstring from a file """
    with file_path.open("r") as f:
        module_ast = ast.parse(f.read())

    docstring = ast.get_docstring(module_ast)
    return docstring

def get_file_list():
    """ Get a list of all python files in the specified directories """
    file_path = Path(__file__).resolve() 
    blech_dir = file_path.parents[2]
    
    # Base dir files
    iter_dirs = [
            blech_dir,
            blech_dir / 'emg',
            blech_dir / 'emg' / 'utils',
            blech_dir / 'utils',
            blech_dir / 'utils' / 'ephys_data',
            blech_dir / 'utils' / 'qa_utils',
            ]

    # Iterate through directories and find all python files
    file_list = []
    for dir in iter_dirs:
        for file in dir.glob('*.py'):
            file_list.append(file)
    sorted_files = sorted(file_list)

    # Remove __init__.py files
    sorted_files = [file for file in sorted_files if '__init__' not in file.name]

    return sorted_files

def create_agent():
    """Create and configure the autogen agents"""

    summary_assistant = AssistantAgent(
        name="summary_assistant",
        llm_config=llm_config,
        system_message="""You are a helpful AI agent for summarizing code
        Keep your summary concise and to the point, but include all relevant information.
        """,
    )

    return summary_assistant

def generate_summary(summary_assistant, file_text):
    """Generate a summary of the code in the file"""
    summary_results = summary_assistant.initiate_chat(
        recipient = summary_assistant,
        message = f"""Summarize the code in the file below to generate a top-level docstring for the module.
Return a summary of the file's contents and the functions it contains.
Follow the format:

General description of the file.
- Bullet points of specific functions. 
- Bullet points of specific functions. 
- Bullet points of specific functions. 
... an so on

Code:
=====
{file_text}
""",
   max_turns = 1,
        )

    llm_docstring = summary_results.chat_history[-1]['content']

    cost = summary_results.cost['usage_excluding_cached_inference']['total_cost']

    return llm_docstring, cost

def check_continue(filename, docstring):
    """Check if the user wants to continue updating the docstring"""
    print(f"Processing {filename}")
    print("Current docstring:")
    print('========================')
    print(docstring)
    print('========================')
    continue_str = input("Continue? (y/n)::: x to exit: ")

    while continue_str not in ['y', 'n', 'x']:
        print("Invalid input. Please enter 'y' or 'n'")
        continue_str = input("Continue? (y/n): ")

    if continue_str == 'y':
        continue_bool = True
        break_bool = False
    elif continue_str == 'n':
        continue_bool = False
        break_bool = False
    elif continue_str == 'x':
        continue_bool = False
        break_bool = True

    return continue_bool, break_bool


def main():

    # Get list of files
    sorted_files = get_file_list()
    
    # Generate summaries for each file
    total_cost = 0
    t = tqdm(sorted_files)
    for file_path in t:
        t.set_description(f'Processing {file_path.name}')
        file_text = file_path.read_text()

        # Get the current docstring
        docstring = docstring_from_file(file_path)

        # Check if the user wants to continue
        if not args.yes:
            continue_bool, break_bool = check_continue(file_path.name, docstring)
            if break_bool:
                break
            if not continue_bool:
                continue

        # Create the autogen agent
        summary_assistant = create_agent()

        # Generate the new docstring
        llm_docstring, cost = generate_summary(summary_assistant, file_text)
        total_cost += cost

        # Search and replace docstring
        if docstring:
            updated_text = file_text.replace(docstring, llm_docstring)
        else:
            updated_text = f'"""\n{llm_docstring}\n"""\n{file_text}'

        # Write updated docstring to file
        with file_path.open("w") as f:
            f.write(updated_text)

        print(f"Updated docstring for {file_path}")

        print(f"Current total cost: {total_cost}")

if __name__ == "__main__":
    main()
