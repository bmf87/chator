from langchain_openai import ChatOpenAI
import streamlit as st

class LModelAccess:
    """
    Model access object for OperRouter model operations.
    """

    
    model_repository = {
        "nvidia": [
            "nvidia/llama-3.3-nemotron-super-49b-v1:free",             # 49B: optimized for advanced reasoning, chat and RAG
        ],
        "meta": [
            "meta-llama/llama-4-scout:free"                             # 109B: 17B active params w/ 16 experts (MoE)
        ],
        "mistral": [
            "mistralai/mistral-small-3.1-24b-instruct:free"             # 24B: variant of Mistral 3.1 optimized for chat and multi-modal tasks
        ],                                          
        "deepseek": [
            "deepseek/deepseek-r1:free",                                # 671B: flagship model 37B active with inference  
            "deepseek/deepseek-prover-v2:free",                         # 685B: fine-tuned for reasoning and RAG      
            "deepseek/deepseek-chat-v3-0324:free"                       # 685B: MoE chat model    
        ]
    }
   

    def __init__(self, app_name, app_dns, api_key=None):
        """
        Initialize the LModelAccess class.
        """
        self.log = st.logger.get_logger(__name__)
        self.app_name = app_name
        self.app_dns = app_dns
        self.api_base_url = "https://openrouter.ai/api/v1"
        self.log.debug("LModelAccess initialized for app: %s", self.app_name)
        if api_key is None:
            self.log.critical("OpenRouter API key was not provided!")
            raise ValueError("OpenRouter API key must be provided!")
        self.api_key = api_key


    
    def get_model_by_provider(self, provider):
        """
        Get the model Ids by provider.

        Args:
            provider (str): The name of the provider (e.g., 'nvidia', 'deepseek').

        Returns:
            list: A list of model Ids for the specified provider.
        """
        if provider in self.model_repository:
            return self.model_repository[provider]
        else:
            raise ValueError(f"Provider '{provider}' not found in model repository.")
    
    def get_all_models(self):
        models = list()
        for key in self.model_repository:
            models.extend(self.get_model_by_provider(key))
        return models 
            
    
    def get_llm(self, model_name, temperature=0.0):
        """
        Get the LLM instance for the specified model name.

        Args:
            model_name (str): The name of the model.
            temperature (float): The temperature for the model. Default is 0.0.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI class.
        """
        llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base_url,
            model_name=model_name,
            default_headers={
                "X-Title": self.app_name,
                "HTTP-Referer": self.app_dns
            },
        )
        return llm