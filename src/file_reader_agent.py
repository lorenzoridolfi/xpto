from typing import Dict, Any, List
from pathlib import Path


class FileReaderAgent:
    """
    Agent responsible for reading files from the 'text' subfolder.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_dir = Path(__file__).parent.parent / "text"

    def read_files(self, filenames: List[str]) -> Dict[str, str]:
        """
        Reads the specified files from the text directory.
        Returns a dictionary mapping filename to file content.
        """
        results = {}
        for fname in filenames:
            file_path = self.text_dir / fname
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    results[fname] = f.read()
            except Exception as e:
                results[fname] = f"ERROR: {str(e)}"
        return results
