import json
import os
from typing import Any, Dict, List
from jsonschema import validate, ValidationError

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config.json')

class UserGeneratorAgent:
    def __init__(self, segment: Dict[str, Any]):
        self.segment = segment

    def generate_user(self) -> Dict[str, Any]:
        # Mock LLM: generate a synthetic user based on segment attributes
        # In real use, replace with LLM call
        return {
            "segmento": self.segment["nome"],
            "perfil": {
                "idade": self.segment["atributos"][0]["valor"],
                "poupanca": self.segment["atributos"][1]["valor"],
                "descricao": self.segment["descricao"]
            }
        }

class ValidatorAgent:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    def validate_user(self, user: Dict[str, Any]) -> (bool, str):
        # For this example, just check required fields
        try:
            # In real use, validate against the schema
            if "segmento" not in user or "perfil" not in user:
                raise ValidationError("Missing required fields")
            return True, ""
        except ValidationError as e:
            return False, str(e)

class ReviewerAgent:
    def review_user(self, user: Dict[str, Any], error: str) -> Dict[str, Any]:
        # Mock review: just add a note
        user["review_note"] = f"Auto-reviewed: {error}"
        return user

class TracedGroupChat:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.trace = []

    def log(self, message: str, data: Any = None):
        entry = {"message": message}
        if data is not None:
            entry["data"] = data
        self.trace.append(entry)

    def save(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.trace, f, indent=2)

class Orchestrator:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        with open(os.path.join(os.path.dirname(__file__), '../' + self.config["input_segment_file"])) as f:
            self.segments = json.load(f)["segmentos"]
        with open(os.path.join(os.path.dirname(__file__), '../' + self.config["synthetic_user_schema_file"])) as f:
            self.schema = json.load(f)
        self.users_per_segment = self.config["users_per_segment"]
        self.output_file = os.path.join(os.path.dirname(__file__), '../' + self.config["output_file"])
        self.tracer = TracedGroupChat(os.path.join(os.path.dirname(__file__), '../' + self.config["log_file"]))

    def run(self):
        all_users = []
        for segment in self.segments:
            self.tracer.log(f"Processing segment: {segment['nome']}")
            generator = UserGeneratorAgent(segment)
            validator = ValidatorAgent(self.schema)
            reviewer = ReviewerAgent()
            segment_users = []
            for i in range(self.users_per_segment):
                user = generator.generate_user()
                self.tracer.log(f"Generated user {i+1}", user)
                valid, error = validator.validate_user(user)
                if not valid:
                    self.tracer.log(f"Validation failed for user {i+1}", error)
                    user = reviewer.review_user(user, error)
                    self.tracer.log(f"User after review {i+1}", user)
                else:
                    self.tracer.log(f"User {i+1} validated successfully")
                segment_users.append(user)
            all_users.extend(segment_users)
        with open(self.output_file, 'w') as f:
            json.dump(all_users, f, indent=2)
        self.tracer.save()

if __name__ == "__main__":
    orchestrator = Orchestrator(CONFIG_PATH)
    orchestrator.run() 