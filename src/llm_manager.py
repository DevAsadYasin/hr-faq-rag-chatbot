import logging
from typing import Optional, List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    LLM_PRIORITY_ORDER,
    LLM_MODEL_OPENROUTER,
    LLM_MODEL_GEMINI,
    LLM_MODEL_OPENAI,
    LLM_TEMPERATURE,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self, priority_order: Optional[List[str]] = None):
        self.priority_order = priority_order or LLM_PRIORITY_ORDER
        self.available_providers = self._detect_available_providers()
        self.llm_instances = {}
        logger.info(f"LLM Manager initialized. Available providers: {self.available_providers}")
        logger.info(f"Priority order: {self.priority_order}")
    
    def _detect_available_providers(self) -> List[str]:
        available = []
        
        if OPENROUTER_API_KEY:
            available.append("openrouter")
            logger.info("✓ OpenRouter API key detected")
        
        if GEMINI_API_KEY:
            available.append("gemini")
            logger.info("✓ Gemini API key detected")
        
        if OPENAI_API_KEY:
            available.append("openai")
            logger.info("✓ OpenAI API key detected")
        
        if not available:
            logger.warning("⚠️  No LLM API keys detected. Answer generation will fail.")
        
        return available
    
    def _get_ordered_providers(self) -> List[str]:
        ordered = []
        for provider in self.priority_order:
            if provider in self.available_providers:
                ordered.append(provider)
        return ordered
    
    def _create_openrouter_llm(self, model: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    
    def _create_gemini_llm(self, model: str, temperature: float) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=GEMINI_API_KEY,
        )
    
    def _create_openai_llm(self, model: str, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=OPENAI_API_KEY,
        )
    
    def _create_llm_instance(self, provider: str, model: str, temperature: float) -> BaseChatModel:
        if provider == "openrouter":
            return self._create_openrouter_llm(model, temperature)
        elif provider == "gemini":
            return self._create_gemini_llm(model, temperature)
        elif provider == "openai":
            return self._create_openai_llm(model, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def get_llm(self, model_override: Optional[str] = None) -> Optional[BaseChatModel]:
        if not self.available_providers:
            logger.error("No LLM providers available. Please set at least one API key.")
            return None
        
        ordered_providers = self._get_ordered_providers()
        
        if not ordered_providers:
            logger.error("No providers available in priority order.")
            return None
        
        if model_override:
            provider = self._detect_provider_from_model(model_override)
            if provider and provider in self.available_providers:
                ordered_providers = [provider] + [p for p in ordered_providers if p != provider]
        
        last_error = None
        for provider in ordered_providers:
            try:
                if provider == "openrouter":
                    model = model_override or LLM_MODEL_OPENROUTER
                elif provider == "gemini":
                    model = model_override or LLM_MODEL_GEMINI
                else:
                    model = model_override or LLM_MODEL_OPENAI
                
                cache_key = f"{provider}:{model}"
                if cache_key not in self.llm_instances:
                    logger.info(f"Creating {provider} LLM instance with model: {model}")
                    self.llm_instances[cache_key] = self._create_llm_instance(
                        provider, model, LLM_TEMPERATURE
                    )
                
                llm = self.llm_instances[cache_key]
                logger.info(f"Using {provider} LLM (model: {model})")
                return llm
                
            except Exception as e:
                logger.warning(f"Failed to create {provider} LLM: {str(e)}")
                last_error = e
                continue
        
        logger.error(f"All LLM providers failed. Last error: {last_error}")
        return None
    
    def _detect_provider_from_model(self, model: str) -> Optional[str]:
        model_lower = model.lower()
        
        if "gemini" in model_lower or "google" in model_lower:
            return "gemini"
        elif any(x in model_lower for x in ["gpt", "openai", "claude", "llama", "mistral"]):
            if model_lower.startswith("gpt-") and "openai" in model_lower:
                return "openai"
            return "openrouter"
        else:
            return None
    
    def invoke_with_fallback(self, chain_dict: Dict[str, Any], query: str) -> Dict[str, Any]:
        ordered_providers = self._get_ordered_providers()
        
        if not ordered_providers:
            return {
                "result": "Error: No LLM providers available. Please set at least one API key (OPENROUTER_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY).",
                "provider_used": None,
                "error": "No providers available"
            }
        
        retriever = chain_dict.get("retriever")
        prompt = chain_dict.get("prompt")
        
        if not retriever or not prompt:
            return {
                "result": "Error: Invalid chain configuration",
                "provider_used": None,
                "error": "Missing retriever or prompt"
            }
        
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        last_error = None
        for provider in ordered_providers:
            try:
                if provider == "openrouter":
                    model = LLM_MODEL_OPENROUTER
                elif provider == "gemini":
                    model = LLM_MODEL_GEMINI
                else:
                    model = LLM_MODEL_OPENAI
                
                llm = self.get_llm()
                if not llm:
                    continue
                
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser
                
                rag_chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                result = rag_chain.invoke({"context": context, "question": query})
                
                logger.info(f"Successfully generated answer using {provider}")
                return {
                    "result": result,
                    "provider_used": provider,
                    "error": None,
                    "source_documents": docs
                }
                
            except Exception as e:
                logger.warning(f"{provider} failed: {str(e)}. Trying next provider...")
                last_error = e
                cache_key = f"{provider}:{model}"
                if cache_key in self.llm_instances:
                    del self.llm_instances[cache_key]
                continue
        
        error_msg = f"All LLM providers failed. Last error: {str(last_error)}"
        logger.error(error_msg)
        return {
            "result": f"I apologize, but I'm unable to generate an answer at this time. All available language models failed. Error: {str(last_error)}. However, I found {len(docs)} relevant document chunks that may help answer your question.",
            "provider_used": None,
            "error": error_msg,
            "source_documents": docs
        }

