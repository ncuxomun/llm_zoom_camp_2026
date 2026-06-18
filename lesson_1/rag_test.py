from openai import OpenAI
from dotenv import load_dotenv
import requests

load_dotenv()

openai_client = OpenAI()

def llm(prompt):
    response = openai_client.responses.create(
        model="gpt-5.4-nano",
        input=prompt
    )
    return response.output_text

# question = "I just discovered the course. Can I join now?"
# # answer = llm(question)
# # print(answer)

# faq_context = """
# I just discovered the course. Can I still join?
# Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions.

# Course: I have registered for the LLM Zoomcamp. When can I expect to receive the confirmation email?
# You don't need it. You're accepted. You can also just start learning and submitting homework (while the form is open) without registering. It is not checked against any registered list. Registration is just to gauge interest before the start date.

# What is the video/zoom link to the stream for the "Office Hours" or live/workshop sessions?
# The zoom link is only published to instructors/presenters/TAs. Students participate via YouTube Live and submit questions to Slido.

# Cloud alternatives with GPU
# Check the quota and reset cycle carefully. Potential options include Google Colab, Kaggle, Databricks.
# """

# prompt = f"""
# Your task is to answer questions from the course participants
# based on the provided context.

# Use the context to find relevant information and provide accurate
# answers. If the answer is not found in the context,
# respond with "I don't know."

# Question:
# {question}

# Context:
# {faq_context}
# """

# answer = llm(prompt)
# print(answer)

# def rag(question):
#     search_results = search(question)
#     user_prompt = build_prompt(question, search_results)
#     return llm(user_prompt)

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()

documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

# print(len(documents))
print(documents[0])