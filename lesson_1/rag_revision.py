from ingest import load_faq_data, build_index
from dotenv import load_dotenv
from openai import OpenAI
from rag_helper import RAGBase

# Load env variables
load_dotenv()
# OpenAI client
client = OpenAI()

# Build index for documents
documents = load_faq_data()
index = build_index(documents)

# Create assistant
instructions = """
You're a course teaching assistant.
Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.
""".strip()

assistant = RAGBase(
    llm_client=client,
    instructions=instructions,
    index=index,
)

# response = assistant.rag("Howw do I run pyjamas locally?")
# print(response)

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

messages = [
    {"role": "user", "content": "I just discovered the course. Can I join it?"}
]

response = client.responses.create(
    model="gpt-5.4-nano",
    input=messages,
    tools=[search_tool],
)

# print(response.output)

import json

call = response.output[0]
args = json.loads(call.arguments)

results = RAGBase(index, client).search(**args)
results_json = json.dumps(results, indent=2)

# Extending the messages // appending history
messages.extend(response.output)
messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": results_json,
})

response = client.responses.create(
    model="gpt-5.4-nano",
    input=messages,
    tools=[search_tool],
)

print(messages)
print("\n\n")
print(response.output_text)