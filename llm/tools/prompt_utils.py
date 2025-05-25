from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

class PromptUtils:
    """
    Utility class for handling prompt templates.
    """
    
    @staticmethod
    def get_prompt(question: str) -> str:
        """
        Generates a prompt for the LLM using the provided user prompt.

        Args:
            question (str): The user's input prompt.

        Returns:
            str: The formatted prompt.
        """
        
        human_input = HumanMessagePromptTemplate.from_template("{user_prompt}")
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
            """
            You are a helpful assistant who always provides a detailed answer. 
            Always start your response beginning with your model name.
            """),
            human_input
        ])
        # Create prompt template
        prompt_tmpl = prompt_template.format_messages(user_prompt=question)
        return prompt_tmpl