# %%
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import psycopg
from tqdm.auto import tqdm

load_dotenv()

from lesson_1.ingest import load_faq_data
from lesson_1.rag_helper import RAGBase

model = SentenceTransformer("all-MiniLM-L6-v2")

q1 = "can i still joing the course after the start date?"
v1 = model.encode(q1)

# %%
# Loading docs
documents = load_faq_data()

# Encoding docs
texts = [doc["question"] + "\n" + doc["answer"] for doc in documents]

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
# X = np.array(vectors)
# 

# Establishing connection with our database
conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

# Creating a table for storing documents
conn.execute(
    """
    DROP TABLE if EXISTS documents
    """
)

conn.execute(
    """
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
    """
)

# Inserting documents with embeddings
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

for doc, vec in tqdm(zip(documents, vectors)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"], vec_to_str(vec)),
    )
conn.commit()

# Searching similar docs
query_str = vec_to_str(v1)

# results = conn.execute(
#     """
#     SELECT course, question, answer,
#         1 - (embedding <=> %s::vector) AS similarity
#     FROM documents
#     ORDER BY embedding <=> %s::vector
#     LIMIT 5
#     """,
#     (query_str, query_str)
# ).fetchall()

# for row in results:
#     print(f"[{row[0]}] {row[1]} (similarity: {row[3]:.4f})")

conn.execute(
    """
    CREATE INDEX on documents
    USING hnsw (embedding vector_cosine_ops)
    """
)

# IN a function
def pgvector_search(query, course="llm-zoomcamp", num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)

    results = conn.execute(
        """
        SELECT course, section, question, answer,
            1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_str, course, query_str, num_results)
    ).fetchall()

    return [
        {"course": row[0], "section": row[1], "question": row[2], "answer": row[3], "similarity": row[-1]}
        for row in results
    ]

# results = pgvector_search(query="How do I join the course")
# print(results)

# results = vs_index.search(query_vector=v1, num_results=5)
# query_vector = model.encode("How do I run Kafka?")
# results = vs_index.search(query_vector, num_results=5)

# vector search
# results = vs_index.search(
#     query_vector=v1,
#     num_results=5,
#     filter_dict={"course": "llm-zoomcamp"},
# )

# print(results)


# Vector based search
class RAGpgVector(RAGBase):
    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        results = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()
    
        return [
            {"course": row[0], "section": row[1], "question": row[2], "answer": row[3]}
            for row in results
        ]

vector_rag = RAGpgVector(embedder=model, conn=conn, llm_client=OpenAI())

res = vector_rag.rag("the program has already begun, can I still sign up?")
print(res)
