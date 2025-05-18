from typing import Dict, Any

class WriterAgent:
    """
    Agent responsible for generating or writing content.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def write(self, content: str, filename: str) -> str:
        """
        Writes the given content to the specified file in the output directory.
        Returns a status message.
        """
        output_dir = self.config.get('output_dir', '.')
        try:
            with open(f"{output_dir}/{filename}", 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {filename}"
        except Exception as e:
            return f"ERROR: {str(e)}"
