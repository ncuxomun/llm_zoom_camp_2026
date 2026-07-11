from ingest import load_faq_data
from evaluation_utils import llm_structured, llm_structured_retry, map_progress, calc_total_price
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel
import json
from tqdm.auto import tqdm
import polars as pl

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

documents = load_faq_data()

documents_llm = []
for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

# print(len(documents_llm))

#
documents = documents_llm

# print(documents[0].get("id"))
# print(documents[0].get("question"))
# print(documents[0].get("answer"))
# 
# Structured output formatting for generated questions
class Questions(BaseModel):
    questions: list[str]

# instructions
data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()

# user prompt
# user_prompt = json.dumps(documents[0])

# messages = [
#     {"role": "developer", "content": data_gen_instructions},
#     {"role": "user", "content": user_prompt}
# ]

# response
# response = openai_client.responses.parse(
#     model="gpt-5.4-nano",
#     input=messages,
#     text_format=Questions,
# )
# print(response.output_parsed)
# # or
# print(response.questions)

# With evals
# result, usage = llm_structured(
#     openai_client,
#     data_gen_instructions,
#     user_prompt,
#     Questions,
# )
# # print(result)
# # print(usage)
# # 
# # 'ground' truth records
# records = []

# for q in result.questions:
#     records.append({"question": q, "document": documents[0]["id"]})

# print(records)

# helpful function
def generate_ground_truth(doc):
    user_prompt = json.dumps(doc)
    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions,
        model="gpt-5.4-nano"
    )

    results = []

    for q in out.questions:
        results.append({"question": q, "document": doc["id"]})

    return results, usage

# first 5 tests
ground_truth = []
usages = []

# sequential
# for doc in tqdm(documents[:5]):
#     records, usage = generate_ground_truth(doc)
#     ground_truth.extend(records)
#     usages.append(usage)

# print(ground_truth)

# parallelized options
with ThreadPoolExecutor(max_workers=5) as executor:
    results = map_progress(executor, documents[:5], generate_ground_truth)

ground_truth = []
usages = []

for records, usage in results:
    ground_truth.extend(records)
    usages.append(usage)

# print(records)
# print(usage)
# 
df_gt = pl.DataFrame(ground_truth)
print(df_gt)

print("Cost: ", calc_total_price(usages))