from ingest import load_faq_data, build_index
from dotenv import load_dotenv
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

load_dotenv()

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

# registering the tool
agent_tools = Tools()
agent_tools.add_tool(search, search_tool)

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    llm_client=OpenAIClient(model="gpt-5.4-nano"),
    developer_prompt=instructions,
    tools=agent_tools,
    chat_interface=chat_interface,
)

# result = runner.loop(
#     prompt = "how do I run ollama locally?",
#     callback=callback,
# )

# print(result.cost)
# print(result.all_messages)
# 
runner.run()