from tqdm import tqdm

# System Prompt
INSTRUCTIONS = """
Your task is to answer questions from the course participats \
based on the provided context. \

Use the content to find relevant information and provide accurate answers.
If the answer is not found in the context, \
respond with "I don't know."
"""

# User prompt
PROMPT_TEMPLATE = """
QUESTION: 
    {question}

CONTEXT:
    {context}
""".strip()

# Base RAG Module
class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
        course: str = "llm-zoomcamp",
        model: str = "gpt-5.4-nano",
    ) -> None:
        """
        Initializes the RAGBase with the given parameters.
        """
        # Initializing the setup
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model

    def search(
        self,
        query: str = "",
        num_results: int = 5,
    ) -> list[dict]:
        """
        Searches the index for the given query and returns the results.
        """
        boost_dict = {
            "question": 3.0,
            "section": 0.5,
        }
        filter_dict = {
            "course": self.course
        }

        return self.index.search(
            query=query,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results,
        )

    def build_context(
        self,
        search_results: list[dict],
    ) -> str:
        """
        Builds the context string from the search results.
        """
        lines = []

        for doc in tqdm(search_results, desc="Building context", colour="yellow"):
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(
        self,
        query: str,
        search_results: list[dict],
    ) -> str:
        """
        Builds the enhanced user prompt from the query and search results.
        """
        context = self.build_context(search_results)
        enhanced_user_prompt = self.prompt_template.format(question = query, context = context)

        return enhanced_user_prompt

    def llm(
        self,
        prompt: str,
    ) -> str:
        """
        Calls the LLM with the given prompt and returns the response.
        """
        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        # Making LLM call
        response = self.llm_client.responses.create(
            model = self.model,
            input = messages,
        )

        return response.output_text

    def rag(
        self,
        query: str,
        num_results: int = 5,
    ) -> str:
        """
        Retrieves augments responses given the dataset
        """
        search_results = self.search(query=query, num_results=num_results)
        prompt = self.build_prompt(query=query, search_results=search_results)
        response = self.llm(prompt=prompt)

        return response