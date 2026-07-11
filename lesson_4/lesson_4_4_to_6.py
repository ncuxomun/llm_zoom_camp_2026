from pathlib import Path
from tqdm.auto import tqdm

import polars as pl
from ingest import build_index, load_faq_data

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

# search function
def text_search(
    query: str = None,
    question_boost: float = 3.0,
    answer_boost: float = 1.0,
    section_boost: float = 0.5,
):
    boost_dict = {
        "question": question_boost,
        "answer": answer_boost,
        "section": section_boost,
    }
    return index.search(query, num_results=5, boost_dict=boost_dict)
    
# 
q = gt_dics[0]
# print(q)

# Retrieve data
doc_id = q["document"]
results = text_search(query = q["question"])

# print(results)
# compare retrieved results
relevance = []

for d in results:
    relevance.append(int(d["id"] == doc_id))

# print(relevance)
# print(pl.DataFrame(results))

def compute_relevance_text(q):
    doc_id = q["document"]
    results = text_search(query=q["question"])

    relevance = []
    for d in results:
        relevance.append(int(d["id"] == doc_id))

    return relevance

# print(q["question"])
# print(compute_relevance_text(q))
# [1, 0, 0, 0, 0]

def compute_relevance_total_text(ground_truth):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance = compute_relevance_text(q)
        relevance_total.append(relevance)

    return relevance_total

# calculate ground truth
gt_sample = gt_dics[:15]
# relevance_total_text = compute_relevance_total_text(gt_sample)

# print(relevance_total_text)

# compute relevance
def compute_relevance(q, search_function):
    doc_id = q["document"]
    results = search_function(query=q["question"])

    relevance = []

    for d in results:
        relevance.append(int(d["id"] == doc_id))

    return relevance

def compute_relevance_total(gt, search_func):
    relevance_total = []

    for q in tqdm(gt):
        relevance = compute_relevance(q, search_func)
        relevance_total.append(relevance)

    return relevance_total

# relevance_total = compute_relevance_total(gt_sample, text_search)
relevance_total = compute_relevance_total(gt_dics, text_search)
# print(relevance_total)
# 
# Hit Rate (aka Recall@k)
def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt += 1
    return cnt/len(relevance)

# hit = hit_rate(relevance_total)
# print(f"Hit Rate: {hit:.3f}")

# MRR
def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance)

# mrr_score = mrr(relevance_total)
# print(f"MMR: {mrr_score:.3f}")

# Putting the eval funcs together
def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    hit = hit_rate(relevance_total)
    mrr_score = mrr(relevance_total)
    return {"hit_rate": hit, "mrr": mrr_score}

# print(evaluate(gt_dics, text_search))
# 
# different boosts

# for boost in [0.5, 1.0, 1.5, 3.0, 5.0, 10.0]:
#     result = evaluate(
#         gt_dics,
#         lambda query, boost=boost: text_search(query=query, question_boost=boost)
#     )
#     print(f"Boost: {boost:.1f} -> Hit Rate: {result['hit_rate']:.3f}, MMR: {result['mrr']:.3f}")
# 
# results = []

# # all settings
# results = []

# for question_boost in [1.0, 2.0, 5.0]:
#     for answer_boost in [1.0, 2.0, 4.0, 10.0]:
#         for section_boost in [0.1, 0.2, 0.5]:
#             print(
#                 f"Evaluating question_boost={question_boost},"
#                 f" answer_boost={answer_boost},"
#                 f" section_boost={section_boost}..."
#             )
#             result = evaluate(
#                 gt_dics,
#                 lambda query, question_boost=question_boost, answer_boost=answer_boost, section_boost=section_boost: text_search(
#                     query,
#                     question_boost,
#                     answer_boost,
#                     section_boost
#                 )
#             )

#             results.append({
#                 "question": question_boost,
#                 "answer": answer_boost,
#                 "section": section_boost,
#                 "hit_rate": result["hit_rate"],
#                 "mrr": result["mrr"],
#             })

# print(pl.DataFrame(results).sort(['mrr'], descending=True).head(10))
# best setup (q: 5.0, a:10, section: 0.2)