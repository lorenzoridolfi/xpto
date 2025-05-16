#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import unittest

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from openai import OpenAI
from src.load_openai import get_openai_config
import requests
from typing import Dict, Any

class TestOpenAI(unittest.TestCase):
    """Test cases for OpenAI integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = get_openai_config()
        self.client = OpenAI(api_key=self.config["api_key"])
    
    def test_openai_connection(self):
        """Test OpenAI connection and basic functionality."""
        try:
            # Test simple completion
            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Bom dia! Como você está?"}
                ],
                temperature=0.7,
                max_tokens=50
            )
            
            # Assert response structure
            self.assertIsNotNone(response.choices)
            self.assertIsNotNone(response.choices[0].message.content)
            self.assertEqual(response.model, self.config["model"])
            self.assertIsNotNone(response.usage)
            
            return {
                "status": "success",
                "message": "Conexão com OpenAI estabelecida com sucesso!",
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_info = {
                "status": "error",
                "message": "",
                "details": ""
            }
            
            if "AuthenticationError" in error_type:
                error_info.update({
                    "message": "Erro de autenticação: A chave da API é inválida ou expirou.",
                    "details": "Verifique se a chave OPENAI_API_KEY no arquivo .env está correta."
                })
            elif "RateLimitError" in error_type:
                error_info.update({
                    "message": "Erro de limite de taxa: Você atingiu o limite de requisições da OpenAI.",
                    "details": "Verifique seus créditos e limites de uso na plataforma OpenAI."
                })
            elif "APIError" in error_type:
                error_info.update({
                    "message": f"Erro na API da OpenAI: {str(e)}",
                    "details": "Pode ser um problema temporário com a API da OpenAI."
                })
            elif "ConnectionError" in error_type:
                error_info.update({
                    "message": "Erro de conexão: Não foi possível conectar à API da OpenAI.",
                    "details": "Verifique sua conexão com a internet e se a API da OpenAI está acessível."
                })
            else:
                error_info.update({
                    "message": f"Erro inesperado: {str(e)}",
                    "details": "Ocorreu um erro não tratado durante o teste."
                })
            
            self.fail(f"{error_info['message']}\n{error_info['details']}")

def run_openai_test():
    """Run OpenAI tests independently."""
    print("Testando conexão com a OpenAI...")
    print("-" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(TestOpenAI('test_openai_connection'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_openai_test()
    sys.exit(0 if success else 1) 