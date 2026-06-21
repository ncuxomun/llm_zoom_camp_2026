#%%
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import numpy as np
from minsearch import VectorSearch
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from lesson_1.ingest import load_faq_data
from lesson_1.rag_helper import RAGBase

model = SentenceTransformer('all-MiniLM-L6-v2')

q1 = "can i still joing the course after the start date?"
v1 = model.encode(q1)

# print(v1.shape)

# d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
# dv = model.encode(d)

# print("Dot product, v1(.) dv = ", v1.dot(dv))

# q2 = "How to install Docker on Windows?"
# v2 = model.encode(q2)

# print("Dot product, v2(.) dv = ", v2.dot(dv))

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

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i+batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

# print(len(vectors))
# print(vectors[0].shape)

# Creating matrix from embedded vectors
X = np.array(vectors)
# print(X.shape)
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
vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)

# results = vindex.search(query_vector=v1, num_results=5)

# Simple search
# results = vindex.search(
#     query_vector=v1,
#     num_results=5,
#     filter_dict={"course": "llm-zoomcamp"},
# ) 

# print(results[0])

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
    index=vindex,
    llm_client = OpenAI()
)

res = vector_rag.rag("the program has already begun, can I still sign up?")
print(res)