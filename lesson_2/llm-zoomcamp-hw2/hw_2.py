from gitsource import GithubRepositoryDataReader
from tqdm.auto import tqdm
from embedder import Embedder

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
texts = [doc["filename"] + "\n" + doc["content"] for doc in documents]

# Question 1
q1_query = "How does approximate nearest neighbor search work?"
emb_model = Embedder()

q1_emb = emb_model.encode(q1_query)
# print(q1_emb)
# answer: -2.058...e-02,....

# Question 2
q2_content = [doc["content"] for doc in documents if 
    "02-vector-search/lessons/07-sqlitesearch-vector.md" in doc["filename"]
    ][0]
q2_emb = emb_model.encode(q2_content)

# print(q1_emb.shape)
# print(q2_emb.shape)
# print("Cosine sim, q2 and q1: ", q2_emb.dot(q1_emb))
# answer:  0.3610702722558969

# Question 3
import numpy as np
from gitsource import chunk_documents
chunks = chunk_documents(documents, size=2000, step=1000)
chunks_cont = [chunk.get("content") for chunk in chunks]

# chunks_emd = emb_model.encode_batch(chunks_cont)

# Scores
# scores = chunks_emd.dot(q1_emb)

# idx = np.argmax(scores)
# print(f"Score: {scores[idx]} @ {idx} position")
# print(f"filename @ {idx} is {chunks[idx].get('filename')}")

# top2 = np.argsort(-scores)[:2]

# for idx in top2:
#     print(scores[idx])
#     print(chunks[idx])
#     print("**--**")
# answer: 02-vector-search/lessons/07-sqlitesearch-vector.md

# Question 4
from minsearch import VectorSearch

# Indexing
vindex = VectorSearch()

# Encoding in batches
batch_size = len(texts) // 3
vectors = []

for i in tqdm(range(0, len(chunks_cont), batch_size)):
    batch = chunks_cont[i:i+batch_size]
    batch_vectors = emb_model.encode_batch(batch)
    vectors.extend(batch_vectors)

# Creating matrix from embedded vectors
X = np.array(vectors)

vindex.fit(X, chunks)

# search results
# v4_emb = emb_model.encode("What metric do we use to evaluate a search engine?")
# results = vindex.search(query_vector=v4_emb, num_results=1)

# print(results)
# answer: '04-evaluation/lessons/05-search-metrics.md'

# Question 5
# search results
q5 = "How do I store vectors in PostgreSQL?"
v5_emb = emb_model.encode(q5)
results_v = vindex.search(query_vector=v5_emb, num_results=5)

print("Vector search results:")
print(sorted([r['filename'] for r in results_v]))
print("=======================================")

# Keyword search
from minsearch import Index

kindex = Index(text_fields=["content"])
kindex.fit(chunks)

results_k = kindex.search(query=q5, num_results=5)

# print("Keyword search results:")
# print(sorted([r['filename'] for r in results_k]))
# print("=======================================")

# Answer
# '02-vector-search/lessons/08-pgvector.md'

# Question 6
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

q6 = "How do I give the model access to tools?"
v6_emb = emb_model.encode(q6)

results_v = vindex.search(query_vector=v6_emb, num_results=5)
results_k = kindex.search(query=q6, num_results=5)

results = rrf([results_v, results_k])

print("Vector search results:")
print(sorted([r['filename'] for r in results_v]))
print("=======================================")

print("Keyword search results:")
print(sorted([r['filename'] for r in results_k]))
print("=======================================")

print("RFP Sorted/Scored results:")
print([r['filename'] for r in results])

# answer: '01-agentic-rag/lessons/13-function-calling.md'