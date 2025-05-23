import json
import os
from typing import Any, Dict, List
from jsonschema import validate, ValidationError

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.json")
AGENT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config_agents.json")
AGENT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "../config_agents_state.json"
)


def load_json(path: str) -> Any:
    """
    Load a JSON file from the given path.
    Args:
        path (str): Path to the JSON file.
    Returns:
        Any: Parsed JSON data.
    """
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: Any):
    """
    Save data as JSON to the given path.
    Args:
        path (str): Path to save the JSON file.
        data (Any): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_and_increment_user_id(state: Dict[str, int], user_id_field: str) -> int:
    """
    Get the current user_id and increment it for the next user.
    Args:
        state (Dict[str, int]): State dictionary containing the user_id.
        user_id_field (str): The field name for the user_id.
    Returns:
        int: The current user_id before incrementing.
    """
    number = state[user_id_field]
    state[user_id_field] += 1
    save_json(AGENT_STATE_PATH, state)
    return number


class UserGeneratorAgent:
    """
    Agent responsible for generating synthetic users for a given segment.
    Assigns a unique sequential user_id to each user.
    """

    def __init__(
        self,
        segment: Dict[str, Any],
        agent_config: Dict[str, Any],
        agent_state: Dict[str, int],
        user_id_field: str,
    ):
        """
        Initialize the UserGeneratorAgent.
        Args:
            segment (Dict[str, Any]): Segment data for user generation.
            agent_config (Dict[str, Any]): Configuration for the agent.
            agent_state (Dict[str, int]): State for user_id tracking.
            user_id_field (str): Field name for the user_id.
        """
        self.segment = segment
        self.config = agent_config
        self.state = agent_state
        self.user_id_field = user_id_field
        self.temperature = self.config["temperature"]
        self.description = self.config["description"]
        self.system_message = self.config["system_message"]

    def generate_user(self) -> Dict[str, Any]:
        """
        Generate a synthetic user for the segment, assigning a unique user_id.
        Returns:
            Dict[str, Any]: The generated synthetic user.
        """
        user_id = get_and_increment_user_id(self.state, self.user_id_field)
        # Mock LLM logic: use segment attributes to create a user profile
        user = {
            self.user_id_field: user_id,
            "segmento": self.segment["nome"],
            "perfil": {
                "idade": self.segment["atributos"][0]["valor"],
                "poupanca": self.segment["atributos"][1]["valor"],
                "descricao": self.segment["descricao"],
            },
        }
        return user


class ValidatorAgent:
    """
    Agent responsible for validating synthetic users against a JSON schema.
    """

    def __init__(self, schema: Dict[str, Any], agent_config: Dict[str, Any]):
        """
        Initialize the ValidatorAgent.
        Args:
            schema (Dict[str, Any]): The JSON schema for validation.
            agent_config (Dict[str, Any]): Configuration for the agent.
        """
        self.schema = schema
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = self.config["description"]
        self.system_message = self.config["system_message"]

    def validate_user(self, user: Dict[str, Any]) -> (bool, str):
        """
        Validate a synthetic user against the schema.
        Args:
            user (Dict[str, Any]): The synthetic user to validate.
        Returns:
            (bool, str): Tuple of (is_valid, error_message)
        """
        try:
            # In real use, validate against the schema
            if "segmento" not in user or "perfil" not in user:
                raise ValidationError("Missing required fields")
            return True, ""
        except ValidationError as e:
            return False, str(e)


class ReviewerAgent:
    """
    Agent responsible for reviewing and suggesting corrections for invalid users.
    """

    def __init__(self, agent_config: Dict[str, Any]):
        """
        Initialize the ReviewerAgent.
        Args:
            agent_config (Dict[str, Any]): Configuration for the agent.
        """
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = self.config["description"]
        self.system_message = self.config["system_message"]

    def review_user(self, user: Dict[str, Any], error: str) -> Dict[str, Any]:
        """
        Review a user that failed validation and add a review note.
        Args:
            user (Dict[str, Any]): The user to review.
            error (str): The validation error message.
        Returns:
            Dict[str, Any]: The reviewed user with a review note.
        """
        user["review_note"] = f"Auto-reviewed: {error}"
        return user


class TracedGroupChat:
    """
    Class responsible for logging all actions and messages for traceability.
    """

    def __init__(self, log_path: str):
        """
        Initialize the TracedGroupChat.
        Args:
            log_path (str): Path to the trace log file.
        """
        self.log_path = log_path
        self.trace = []

    def log(self, message: str, data: Any = None):
        """
        Log a message and optional data to the trace.
        Args:
            message (str): The log message.
            data (Any, optional): Additional data to log.
        """
        entry = {"message": message}
        if data is not None:
            entry["data"] = data
        self.trace.append(entry)

    def save(self):
        """
        Save the trace log to the log file.
        """
        with open(self.log_path, "w") as f:
            json.dump(self.trace, f, indent=2)


class Orchestrator:
    """
    Orchestrates the synthetic user generation workflow, coordinating all agents.
    """

    def __init__(self, config_path: str, agent_config_path: str, agent_state_path: str):
        """
        Initialize the Orchestrator.
        Args:
            config_path (str): Path to the main config file.
            agent_config_path (str): Path to the agent config file.
            agent_state_path (str): Path to the agent state file.
        """
        self.config = load_json(config_path)
        self.agent_config = load_json(agent_config_path)
        self.agent_state = load_json(agent_state_path)
        self.user_id_field = self.agent_config["user_id_field"]
        # Load segments and schema
        with open(
            os.path.join(
                os.path.dirname(__file__), "../" + self.config["input_segment_file"]
            )
        ) as f:
            self.segments = json.load(f)["segmentos"]
        with open(
            os.path.join(
                os.path.dirname(__file__),
                "../" + self.config["synthetic_user_schema_file"],
            )
        ) as f:
            self.schema = json.load(f)
        self.users_per_segment = self.config["users_per_segment"]
        self.output_file = os.path.join(
            os.path.dirname(__file__), "../" + self.config["output_file"]
        )
        self.tracer = TracedGroupChat(
            os.path.join(os.path.dirname(__file__), "../" + self.config["log_file"])
        )

    def run(self):
        """
        Run the synthetic user generation workflow for all segments.
        Generates users, validates, reviews if needed, and logs all actions.
        """
        all_users = []  # List to collect all generated users
        for segment in self.segments:
            self.tracer.log(f"Processing segment: {segment['nome']}")
            # Instantiate agents for this segment
            generator = UserGeneratorAgent(
                segment,
                self.agent_config["UserGeneratorAgent"],
                self.agent_state,
                self.user_id_field,
            )
            validator = ValidatorAgent(self.schema, self.agent_config["ValidatorAgent"])
            reviewer = ReviewerAgent(self.agent_config["ReviewerAgent"])
            segment_users = []  # Users for this segment
            for i in range(self.users_per_segment):
                # Generate a user
                user = generator.generate_user()
                self.tracer.log(f"Generated user {i+1}", user)
                # Validate the user
                valid, error = validator.validate_user(user)
                if not valid:
                    self.tracer.log(f"Validation failed for user {i+1}", error)
                    # Review and correct the user
                    user = reviewer.review_user(user, error)
                    self.tracer.log(f"User after review {i+1}", user)
                else:
                    self.tracer.log(f"User {i+1} validated successfully")
                segment_users.append(user)
            all_users.extend(segment_users)
        # Save all users to the output file
        with open(self.output_file, "w") as f:
            json.dump(all_users, f, indent=2)
        # Save the trace log
        self.tracer.save()
        # Save updated user_id state
        save_json(AGENT_STATE_PATH, self.agent_state)


if __name__ == "__main__":
    # Entry point for running the orchestrator
    orchestrator = Orchestrator(CONFIG_PATH, AGENT_CONFIG_PATH, AGENT_STATE_PATH)
    orchestrator.run()
