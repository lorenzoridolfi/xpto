from typing import Dict, Any


class QualityAgent:
    """
    Agent responsible for assessing the quality of text content.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def assess_quality(self, content: str) -> str:
        """
        Dummy quality assessment. Returns 'good' if content length > 10, else 'poor'.
        """
        return "good" if len(content) > 10 else "poor"
