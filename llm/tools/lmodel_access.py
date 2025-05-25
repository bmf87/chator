from langchain_openai import ChatOpenAI

class LModelAccess:
    """
    Model access object for OperRouter model operations.
    """

    
    model_repository = {
        "nvidia": [
            "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",             # 253B: optimized for advanced reasoning, chat and RAG
        ],
        "mistral": [
            "mistralai/mistral-small-3.1-24b-instruct:free"             # 24B: variant of Mistral 3.1 optimized for chat and multi-modal tasks
        ],
        "meta":[
            "meta-llama/llama-3.1-405b:free",                           # 405B: base pre-trained flagship model
        ],                                                      
        "deepseek": [
            "deepseek/deepseek-r1:free",                                # 671B: flagship model    
            "deepseek/deepseek-prover-v2:free",                         # 685B: fine-tuned for reasoning and RAG      
            "deepseek/deepseek-chat-v3-0324:free"                       # 685B: mixture of experts model    
        ]
    }
   

    def __init__(self, api_key=None):
        """
        Initialize the LModelAccess class.
        """
        self.api_base_url = "https://openrouter.ai/api/v1"
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
            temperature (float): The temperature for the model.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI class.
        """
        llm = ChatOpenAI(
            temperature=temperature,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base_url,
            model_name=model_name
        )
        return llm

