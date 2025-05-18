from typing import Dict, Any

class VerifierAgent:
    """
    Agent responsible for verifying the correctness of content or results.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def verify(self, content: str) -> bool:
        """
        Dummy verification logic. Returns True if content is non-empty.
        """
        return bool(content and content.strip())
