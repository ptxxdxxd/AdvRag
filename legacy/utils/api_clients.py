"""
API Client for multiple LLM providers
Supports OpenAI, Anthropic, and other major LLM APIs
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import aiohttp
import requests

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API clients"""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60


class LLMAPIClient:
    """Unified client for multiple LLM providers"""
    
    def __init__(self, configs: List[APIConfig]):
        """
        Initialize with multiple API configurations
        
        Args:
            configs: List of API configurations for different providers
        """
        self.configs = configs
        self.providers = {}
        
        # Initialize available providers
        for config in configs:
            if config.provider.lower() == "openai":
                self.providers["openai"] = OpenAIClient(config)
            elif config.provider.lower() == "anthropic":
                self.providers["anthropic"] = AnthropicClient(config)
            elif config.provider.lower() == "huggingface":
                self.providers["huggingface"] = HuggingFaceClient(config)
            else:
                logger.warning(f"Unknown provider: {config.provider}")
    
    def get_client(self, provider: str):
        """Get client for specific provider"""
        return self.providers.get(provider.lower())
    
    async def generate_with_fallback(
        self, 
        prompt: str, 
        providers: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response with fallback between providers
        
        Args:
            prompt: Input prompt
            providers: List of providers to try in order
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with response and metadata
        """
        if providers is None:
            providers = list(self.providers.keys())
        
        for provider in providers:
            client = self.get_client(provider)
            if client:
                try:
                    response = await client.generate(prompt, **kwargs)
                    response["provider"] = provider
                    return response
                except Exception as e:
                    logger.warning(f"Provider {provider} failed: {str(e)}")
                    continue
        
        raise Exception("All providers failed to generate response")


class OpenAIClient:
    """OpenAI API client"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using OpenAI API"""
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["choices"][0]["message"]["content"],
                        "usage": data.get("usage", {}),
                        "model": data.get("model", self.config.model),
                        "provider": "openai"
                    }
                else:
                    error_data = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_data}")


class AnthropicClient:
    """Anthropic API client"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Anthropic API"""
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        payload = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data["content"][0]["text"],
                        "usage": data.get("usage", {}),
                        "model": data.get("model", self.config.model),
                        "provider": "anthropic"
                    }
                else:
                    error_data = await response.text()
                    raise Exception(f"Anthropic API error: {response.status} - {error_data}")


class HuggingFaceClient:
    """Hugging Face API client"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.base_url = config.base_url or "https://api-inference.huggingface.co/models"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using Hugging Face API"""
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Prepare inputs for text generation
        inputs = prompt
        
        payload = {
            "inputs": inputs,
            "parameters": {
                "max_length": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
                "do_sample": True,
                **kwargs
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/{self.config.model}",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    # Handle different response formats
                    if isinstance(data, list) and len(data) > 0:
                        response_text = data[0].get("generated_text", "")
                    elif isinstance(data, dict):
                        response_text = data.get("generated_text", "")
                    else:
                        response_text = str(data)
                    
                    return {
                        "response": response_text,
                        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_text.split())},
                        "model": self.config.model,
                        "provider": "huggingface"
                    }
                else:
                    error_data = await response.text()
                    raise Exception(f"Hugging Face API error: {response.status} - {error_data}")


class LocalModelClient:
    """Client for local models using libraries like Ollama"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434/api"
    
    async def generate(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using local model"""
        
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            },
            **kwargs
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout * 2)  # Local models may be slower
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "response": data.get("response", ""),
                        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": data.get("eval_count", 0)},
                        "model": self.config.model,
                        "provider": "local"
                    }
                else:
                    error_data = await response.text()
                    raise Exception(f"Local model API error: {response.status} - {error_data}")


def create_api_client_from_env():
    """Create API client from environment variables"""
    configs = []
    
    # OpenAI
    if os.getenv("OPENAI_API_KEY"):
        configs.append(APIConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        ))
    
    # Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        configs.append(APIConfig(
            provider="anthropic",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        ))
    
    # Hugging Face
    if os.getenv("HUGGINGFACE_API_KEY"):
        configs.append(APIConfig(
            provider="huggingface",
            api_key=os.getenv("HUGGINGFACE_API_KEY"),
            model=os.getenv("HUGGINGFACE_MODEL", "microsoft/DialoGPT-medium")
        ))
    
    if configs:
        return LLMAPIClient(configs)
    else:
        logger.warning("No API keys found in environment variables")
        return None