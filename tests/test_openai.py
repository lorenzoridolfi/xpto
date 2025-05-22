"""
OpenAI Integration Tests

This module provides test cases for OpenAI API integration. It includes:
- Connection testing
- Basic functionality verification
- Error handling and reporting
- Independent test execution

The tests verify the proper configuration and functionality of the OpenAI integration,
ensuring reliable operation of the system.
"""

#!/usr/bin/env python3

import sys
import unittest
from typing import Dict, Optional, TypedDict
from openai import OpenAI
from src.load_openai import get_openai_config

# Custom Exceptions
class TestError(Exception):
    """Base exception for test-related errors."""
    pass

class TestConfigurationError(TestError):
    """Raised when test configuration is invalid."""
    pass

class TestExecutionError(TestError):
    """Raised when test execution fails."""
    pass

# Type Definitions
class TestResult(TypedDict):
    """Type definition for test results."""
    status: str
    message: str
    response: Optional[str]
    model: Optional[str]
    usage: Optional[Dict[str, int]]

class ErrorInfo(TypedDict):
    """Type definition for error information."""
    status: str
    message: str
    details: str

class TestOpenAI(unittest.TestCase):
    """Test cases for OpenAI integration.
    
    This class provides comprehensive test cases for verifying the OpenAI API
    integration, including connection testing, basic functionality verification,
    and error handling.
    """
    
    def setUp(self) -> None:
        """Set up test environment.
        
        Raises:
            TestConfigurationError: If test setup fails
        """
        try:
            self.config = get_openai_config()
            self.client = OpenAI(api_key=self.config["api_key"])
        except Exception as e:
            raise TestConfigurationError(f"Failed to setup test environment: {str(e)}")
    
    def test_openai_connection(self) -> TestResult:
        """Test OpenAI connection and basic functionality.
        
        Returns:
            Dictionary containing test results
            
        Raises:
            TestExecutionError: If test execution fails
        """
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
            self.assertTrue(response.model.startswith(self.config["model"]))
            self.assertIsNotNone(response.usage)
            
            return {
                "status": "success",
                "message": "Conexão com OpenAI estabelecida com sucesso!",
                "response": response.choices[0].message.content,
                "model": response.model,
                "usage": response.usage
            }
            
        except Exception as e:
            error_info = self._handle_test_error(e)
            self.fail(f"{error_info['message']}\n{error_info['details']}")
            raise TestExecutionError(f"Test execution failed: {str(e)}")
    
    def _handle_test_error(self, error: Exception) -> ErrorInfo:
        """Handle test errors and provide detailed information.
        
        Args:
            error: Exception that occurred
            
        Returns:
            Dictionary containing error information
        """
        error_type = type(error).__name__
        error_info: ErrorInfo = {
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
                "message": f"Erro na API da OpenAI: {str(error)}",
                "details": "Pode ser um problema temporário com a API da OpenAI."
            })
        elif "ConnectionError" in error_type:
            error_info.update({
                "message": "Erro de conexão: Não foi possível conectar à API da OpenAI.",
                "details": "Verifique sua conexão com a internet e se a API da OpenAI está acessível."
            })
        else:
            error_info.update({
                "message": f"Erro inesperado: {str(error)}",
                "details": "Ocorreu um erro não tratado durante o teste."
            })
        
        return error_info

def run_openai_test() -> bool:
    """Run OpenAI tests independently.
    
    Returns:
        True if all tests passed, False otherwise
        
    Raises:
        TestExecutionError: If test execution fails
    """
    try:
        print("Testando conexão com a OpenAI...")
        print("-" * 50)
        
        # Create test suite
        suite = unittest.TestSuite()
        suite.addTest(TestOpenAI('test_openai_connection'))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except Exception as e:
        raise TestExecutionError(f"Failed to run OpenAI tests: {str(e)}")

if __name__ == "__main__":
    try:
        success = run_openai_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        sys.exit(1) 