from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from ingest import build_index, load_faq_data
from evaluation_utils import RAGWithUsage, map_progress

load_dotenv()
openai_client = OpenAI()

data_paths = Path("lesson_4/data")

df_gt = pl.read_csv(data_paths / "ground_truth-new.csv")
gt_dics = df_gt.to_dicts()

# load and build index
documents = load_faq_data()
documents_llm = []

for doc in documents:
    if doc['course'] == "llm-zoomcamp":
        documents_llm.append(doc)
#
documents = documents_llm
index = build_index(documents)

#doc index
doc_idx = {}

for doc in documents:
    doc_idx[doc['id']] = doc

# assistant
assistant = RAGWithUsage(
    index=index,
    llm_client=openai_client,
)

# 
rec = gt_dics[0]
question = rec['question']

# answer_llm = assistant.rag(question)
# print(answer_llm)
# print(assistant.total_cost())
# 
# doc_id = rec["document"]
# original_doc = doc_idx[doc_id]
# answer_orig = original_doc["answer"]

# rag_result = {
#     "question": question,
#     "answer_llm": answer_llm,
#     "answer_orig": answer_orig,
#     "document": doc_id,
# }

# print(rag_result)

# reusable function
def generate_rag_answer(rec):
    question = rec["question"]
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]
    
    answer_orig = original_doc["answer"]
    answer_llm = assistant.rag(question)

    return {
        "question": question,
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
    }

# answer_record = generate_rag_answer(rec)
# print(answer_record)
answers = []

# with ThreadPoolExecutor(max_workers=6) as executor:
#     results = map_progress(executor, gt_dics[:30], generate_rag_answer)
    
# for result in results:
#     answers.append(result)

# print(assistant.total_cost())

# df_answers = pl.DataFrame(answers)
# df_answers.write_csv(data_paths / "rag-answers-new.csv_nano.csv")
print(documents[0])