from gitsource import GithubRepositoryDataReader
from pydantic import BaseModel
from evaluation_utils import llm_structured
import json
from minsearch import VectorSearch, Index
from embedder import Embedder
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI
import polars as pl
from tqdm.auto import tqdm

from gitsource import chunk_documents

load_dotenv()
openai_client = OpenAI()

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

############## Q1 ##############
data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()

# Structured output formatting for generated questions
class Questions(BaseModel):
    questions: list[str]

# user prompt
q1_tests = [
    "01-agentic-rag/lessons/01-intro.md",
    "01-agentic-rag/lessons/02-environment.md",
    "01-agentic-rag/lessons/03-rag.md"
]
user_prompt = json.dumps(documents[0])

messages = [
    {"role": "developer", "content": data_gen_instructions},
    {"role": "user", "content": user_prompt}
]

# for f in q1_tests:
#     result, usage = llm_structured(
#         openai_client,
#         data_gen_instructions,
#         user_prompt,
#         Questions
#     )

#     print(f"{f} usage: {usage}")
# Answer Q1: ~ 1200

############## Q2 ##############
chunks = chunk_documents(documents, size=2000, step=1000)

emb_model = Embedder(path="lesson_4/models/Xenova/all-MiniLM-L6-v2")

def text_search(q, num_results=5):
    kindex = Index(text_fields=["content"], keyword_fields=["filename"])
    kindex.fit(chunks)
    
    results_k = kindex.search(query=q, num_results=num_results)
    return results_k

# Encoding in batches
vindex = VectorSearch(keyword_fields=["filename"])
batch_size = len(chunks) // 3
vectors = []

chunks_cont = [chunk.get("content") for chunk in chunks]
   
for i in tqdm(range(0, len(chunks_cont), batch_size)):
    batch = chunks_cont[i:i+batch_size]
    batch_vectors = emb_model.encode_batch(batch)
    vectors.extend(batch_vectors)

# Creating matrix from embedded vectors
X = np.array(vectors)
   
vindex.fit(X, chunks)

def vector_search(q, num_results=5):
    results_v = vindex.search(query_vector=emb_model.encode(q), num_results=num_results)
    return results_v

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)

ground_truth = pl.read_csv("lesson_4/data/ground-truth.csv").rows(named=True)
q = ground_truth[0]["question"]

# print(q)
# print(text_search(q, num_results=1))

# Answer: 01-agentic-rag/lessons/03-rag.md
############## Q3 ##############
# print(vector_search(q, num_results=1))
# Answer: '01-agentic-rag/lessons/01-intro.md'
# 

############## Q4 ##############
# # compute relevance
def compute_relevance(q, search_function):
    doc_id = q["filename"]
    results = search_function(q["question"])

    relevance = []

    for d in results:
        relevance.append(int(d["filename"] == doc_id))

    return relevance

def compute_relevance_total(gt, search_func):
    relevance_total = []

    for q in tqdm(gt):
        relevance = compute_relevance(q, search_func)
        relevance_total.append(relevance)

    return relevance_total

# 
# Hit Rate (aka Recall@k)
def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt += 1
    return cnt/len(relevance)

# MRR
def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance)

# Putting the eval funcs together
def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    hit = hit_rate(relevance_total)
    mrr_score = mrr(relevance_total)
    return {"hit_rate": hit, "mrr": mrr_score}

# print(evaluate(ground_truth, text_search))
# answer: 0.76
# 
############## Q5 ##############
# print(evaluate(ground_truth, vector_search))
# answer: 0.55
# 
for k in [1, 50, 100, 200]:
    result = evaluate(ground_truth, lambda query, k=k: hybrid_search(query, k=k))
    print(f"k={k}: hit_rate={result['hit_rate']}, mrr={result['mrr']}")

# ans: k = 1