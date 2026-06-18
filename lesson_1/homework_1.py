from gitsource import GithubRepositoryDataReader
from minsearch import Index
from dotenv import load_dotenv
from gitsource import chunk_documents

load_dotenv()

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()
# print(len(files))
# print(files[15

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

# print(len(documents))
# print(documents[0].keys())

chunks = chunk_documents(documents, size=2000, step=1000)
# print(chunks[-1].keys())
# print(len(chunks))

# Function to build 'minsearch' Index
def build_index(documents: list[dict], text_field: list[str], keywords: list[str]) -> Index:
    """
    Builds a 'minsearch' Index from a list of documents.
    """
    
    index = Index(
        text_fields=text_field,
        keyword_fields=keywords,
    )
    index.fit(documents)
    return index

# query = "How does the agentic loop keep calling the model until it stops?"
# search_index = build_index(documents, ["content"], ["filename"])
search_index = build_index(chunks, ["content"], ["filename"])

# results = search_index.search(query, num_results=1)
# print(results[0].get("filename"))
from rag_helper_hw import RAGBase

# rag_setup = RAGBase(
#     index = search_index,
#     model="gpt-5.4-mini",
# )

# res_1, res_2 = rag_setup.rag(query=query, num_results=5)

# print(res_1)
# print(res_2)
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return search_index.search(
        query,
        num_results=5,
        output_ids=True
    )

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback

instructions = """
You're a course teaching assistant. Answer the student's question using the search tool. \
Make multiple searches with different keywords before answering.
""".strip()

query = "How does the agentic loop work, and how is it different from plain RAG?"

agent_tools = Tools()
agent_tools.add_tool(search)

chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    llm_client=OpenAIClient(model="gpt-5.4-nano"),
    developer_prompt=instructions,
    tools=agent_tools,
    chat_interface=chat_interface,
)

result = runner.loop(
    prompt = query,
    callback=callback,
)

print(result.cost)
print(result.all_messages)