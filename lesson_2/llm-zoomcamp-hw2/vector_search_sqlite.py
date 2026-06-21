#%%
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from sqlitesearch import VectorSearchIndex
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from lesson_1.ingest import load_faq_data
from lesson_1.rag_helper import RAGBase

model = SentenceTransformer('all-MiniLM-L6-v2')

q1 = "can i still joing the course after the start date?"
v1 = model.encode(q1)

#%%
# Loading docs
documents = load_faq_data()

# Encoding docs
texts = []

for doc in documents:
    text = doc["question"] + "\n" + doc["answer"]
    texts.append(text)

# Encoding in batches
batch_size = 50
vectors = []

# for i in tqdm(range(0, len(texts), batch_size)):
#     batch = texts[i:i+batch_size]
#     batch_vectors = model.encode(batch)
#     vectors.extend(batch_vectors)

# print(len(vectors))
# print(vectors[0].shape)

# Creating matrix from embedded vectors
# X = np.array(vectors)

# print(X.shape)
# 
#%%
# scores = X.dot(v1)
# idx = np.argmax(scores)

# print(f"Score: {scores[idx]} @ {idx} position")

# top5 = np.argsort(-scores)[:5]

# for idx in top5:
#     print(scores[idx])
#     print(documents[idx])
#     print(22 * "**--**")
# 
# Indexing
vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="databases/faq_vslite.db",
    seed=42,
)
# vs_index.fit(vectors=X, payload=documents)

# results = vs_index.search(query_vector=v1, num_results=5)
query_vector = model.encode("How do I run Kafka?")
# results = vs_index.search(query_vector, num_results=5)

# vector search
# results = vs_index.search(
#     query_vector=v1,
#     num_results=5,
#     filter_dict={"course": "llm-zoomcamp"},
# ) 

# print(results)

# Vector based search
class RAGVector(RAGBase):
    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector=query_vector,
            num_results=num_results,
            filter_dict=filter_dict,
        )

vector_rag = RAGVector(
    embedder=model, 
    index=vs_index,
    llm_client = OpenAI()
)

res = vector_rag.rag("the program has already begun, can I still sign up?")
print(res)

vs_index.close()