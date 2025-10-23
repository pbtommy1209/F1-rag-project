#!/usr/bin/env python3
"""
Ollama Client for RAG System
Handles communication with Ollama API for embeddings and chat
"""

import requests
import json
from typing import List, Dict, Any


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.embed_url = f"{self.base_url}/api/embeddings"
        self.chat_url = f"{self.base_url}/api/chat"
    
    def embed_batch(self, model: str, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.embed_url,
                    json={
                        "model": model,
                        "prompt": text
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result.get("embedding", []))
                else:
                    print(f"❌ Embedding failed: {response.status_code}")
                    embeddings.append([])
            except Exception as e:
                print(f"❌ Embedding error: {e}")
                embeddings.append([])
        return embeddings
    
    def embed_single(self, model: str, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = requests.post(
                self.embed_url,
                json={
                    "model": model,
                    "prompt": text
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                print(f"❌ Embedding failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return []
    
    def chat(self, model: str, messages: List[Dict[str, str]]) -> str:
        """Send chat request to Ollama"""
        try:
            response = requests.post(
                self.chat_url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                print(f"❌ Chat failed: {response.status_code}")
                return "Error: Chat request failed"
        except Exception as e:
            print(f"❌ Chat error: {e}")
            return f"Error: {str(e)}"
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                result = response.json()
                return [model["name"] for model in result.get("models", [])]
            return []
        except:
            return []
