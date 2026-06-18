from ingest import load_faq_data, build_index
from dotenv import load_dotenv
from openai import OpenAI
from rag_helper import RAGBase
import json

# Load env variables
load_dotenv()
# OpenAI client
client = OpenAI()

# Build index for documents
documents = load_faq_data()
index = build_index(documents)

instructions = """
You're a course teaching assistant.
You're given a question from a course and your task is to answer it.

If you want to look up information, use the `search` function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results \
and then perform more searches.

At the end, ask if there are other areas that the user wants to explore.
""".strip()

# function-call helper
def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = RAGBase(index, client).search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

question = "I just discovered the course. Can I join it?"

def agent_loop(
    instructions,
    question,
    model="gpt-5.4-nano"
):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question},
    ]
    
    iter = 1
    
    while True:
        print(f"Iteration #{iter}...")
        has_function_calls = False
        
        response = client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool],
        )
    
        messages.extend(response.output)
    
        for item in response.output:
            if item.type == "function_call":
                print("function_call: ", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True
            elif item.type == "message":
                print("Assistant: ")
                last_answer = item.content[0].text
                print(item.content[0].text)
    
        iter += 1
        if has_function_calls == False:
            break
    return last_answer

def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )

agent_loop(instructions, "How to cook tom yum?")