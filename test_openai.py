#!/usr/bin/env python3

import sys
import openai
from load_openai import get_openai_config
import requests
from typing import Dict, Any

def test_openai_connection() -> Dict[str, Any]:
    """
    Test OpenAI connection and return detailed status information.
    Returns a dictionary with test results and any error messages.
    """
    try:
        # Get OpenAI configuration
        config = get_openai_config()
        
        # Configure OpenAI client
        openai.api_key = config["api_key"]
        
        # Test simple completion
        response = openai.ChatCompletion.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Bom dia! Como você está?"}
            ],
            temperature=0.7,
            max_tokens=50
        )
        
        return {
            "status": "success",
            "message": "Conexão com OpenAI estabelecida com sucesso!",
            "response": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage
        }
        
    except openai.error.AuthenticationError:
        return {
            "status": "error",
            "message": "Erro de autenticação: A chave da API é inválida ou expirou.",
            "details": "Verifique se a chave OPENAI_API_KEY no arquivo .env está correta."
        }
        
    except openai.error.RateLimitError:
        return {
            "status": "error",
            "message": "Erro de limite de taxa: Você atingiu o limite de requisições da OpenAI.",
            "details": "Verifique seus créditos e limites de uso na plataforma OpenAI."
        }
        
    except openai.error.APIError as e:
        return {
            "status": "error",
            "message": f"Erro na API da OpenAI: {str(e)}",
            "details": "Pode ser um problema temporário com a API da OpenAI."
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "message": "Erro de conexão: Não foi possível conectar à API da OpenAI.",
            "details": "Verifique sua conexão com a internet e se a API da OpenAI está acessível."
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro inesperado: {str(e)}",
            "details": "Ocorreu um erro não tratado durante o teste."
        }

def main():
    """Main function to run the OpenAI connection test."""
    print("Testando conexão com a OpenAI...")
    print("-" * 50)
    
    result = test_openai_connection()
    
    if result["status"] == "success":
        print("\n✅", result["message"])
        print("\nResposta do modelo:")
        print("-" * 50)
        print(result["response"])
        print("-" * 50)
        print(f"\nModelo usado: {result['model']}")
        print(f"Tokens usados: {result['usage'].total_tokens}")
    else:
        print("\n❌", result["message"])
        print("\nDetalhes do erro:")
        print("-" * 50)
        print(result["details"])
        sys.exit(1)

if __name__ == "__main__":
    main() 