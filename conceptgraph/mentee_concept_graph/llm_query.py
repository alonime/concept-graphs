from openai import OpenAI
import os
import base64

from PIL import Image
import numpy as np

import ast
import json
import re


def extract_dict_of_tuples(text: str):
    text = text.replace('\n', ' ')
    pattern = r'\{.*?\}'
    
    match = re.search(pattern, text)
    if match:
        # Extract the matched string
        list_str = match.group(0)
        try:
            # Convert the string to a list of tuples
            result =json.loads(list_str)
            result['relevant_objects'] = [int(idx) for idx in result['relevant_objects']]
            result['final_relevant_objects'] = [int(idx) for idx in result['final_relevant_objects']]
            return result
        
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            print("Found string cannot be converted to a list of tuples.")
            return []
    else:
        # No matching pattern found
        print("No list of tuples found in the text.")
        return []


def llm_retrive_object(client: OpenAI, user_query, objects_json):

    system_prompt = """The input to the model is a 3D scene described in a JSON format. Each entry in the JSON describes one object in the scene, each ibject have a list of captions.
                        Once you have parsed the JSON and are ready to answer questions about the scene. The user will then begin to ask questions, 
                        and the task is to answer various user queries about the 3D scene. For each user question, respond with a JSON dictionary with the following
                        fields: 
                        1. "inferred_query": your interpretaion of the user query in a succinct form
                        2. "relevant_objects": list of relevant objects, with the index of entry in the input JSON for the user query (if applicable)
                        3. "query_achievable": whether or not the userspecified query is achievable using the objects and descriptions provided in the 3D scene.
                        4. "final_relevant_objects": A final list of objects relevant to the user-specified task. 
                        As much as possible, sort all objects in this list such that the most relevant object is listed first, followed by the second most relevant, and so on.
                        5. "explanation": A brief explanation of what the most relevant object(s) is(are), and how they  achieve the user-specified task. 
                    """
       
    
    llm_answer = []
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": objects_json
                    },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )
        
        llm_answer_str = response.choices[0].message.content
        
        llm_answer = extract_dict_of_tuples(llm_answer_str)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(f"Setting vlm_answer to an empty list.")
        llm_answer = []
    
    
    return llm_answer