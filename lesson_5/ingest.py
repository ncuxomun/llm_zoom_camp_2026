import requests
from tqdm import tqdm
from minsearch import Index

# Function to load the FAQ dataset
def load_faq_data() -> list[dict]:
    """
    Loads the FAQ dataset from the specified URL and collects documents from each course.

    Returns:
        list[dict]: A list of dictionaries representing the FAQ data.
    """
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(url = docs_url)

    # Data in raw format
    courses_raw = response.json()

    # Collecting documents
    documents = []
    url_prefix = "https://datatalks.club/faq/"

    for course in tqdm(courses_raw, desc="Collecting data", colour="magenta"):
        course_url = f"{url_prefix}{course['path']}"
        course_response = requests.get(url = course_url)
        course_response.raise_for_status()
        course_faq_data = course_response.json()

        documents.extend(course_faq_data)

    return documents
        
# Function to build 'minsearch' Index
def build_index(documents: list[dict]) -> Index:
    """
    Builds a 'minsearch' Index from the given documents.

    Args:
        documents (list[dict]): A list of dictionaries representing the FAQ data.

    Returns:
        Index: The built 'minsearch' Index.
    """
    index = Index(
        text_fields=["question", "section",  "answer"],
        keyword_fields=["course"],
    )
    index.fit(documents)
    return index