import json
import os
from typing import Any, Dict, List
from jsonschema import validate, ValidationError
import openai
from dotenv import load_dotenv
from pydantic import ValidationError as PydanticValidationError
from tradeshow.src.pydantic_schema import SyntheticUser, CriticOutput

# --- LLM and API Key Handling (as in update_manifest) ---
# Load .env from project root if present
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(DOTENV_PATH):
    load_dotenv(dotenv_path=DOTENV_PATH)

# Read OPENAI_API_KEY from environment
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment!")
# Set for downstream libraries
os.environ["OPENAI_API_KEY"] = openai_api_key
openai.api_key = openai_api_key

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.json")
AGENT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config_agents.json")
AGENT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "../config_agents_state.json"
)
SEGMENTS_PATH = os.path.join(os.path.dirname(__file__), "../input/segments.json")
SEGMENT_SCHEMA_PATH = os.path.join(
    os.path.dirname(__file__), "../schema/segmets_schema.json"
)
AGENTS_UPDATE_PATH = os.path.join(
    os.path.dirname(__file__), "../other/agents_update.json"
)

# --- Load agent updates ---
with open(AGENTS_UPDATE_PATH) as f:
    AGENTS_UPDATE = json.load(f)


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


def validate_segments_schema():
    """
    Validate the segments.json file against the segment schema. Raises an error if invalid.
    """
    schema = load_json(SEGMENT_SCHEMA_PATH)
    segments = load_json(SEGMENTS_PATH)
    try:
        validate(instance=segments, schema=schema)
    except ValidationError as e:
        raise RuntimeError(f"segments.json validation error: {e.message}")


# Validate segments.json at import/run time
validate_segments_schema()


class TracedGroupChat:
    """
    Class responsible for logging all actions and messages for traceability.
    Each log entry includes agent metadata and activity context.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.trace = []

    def log(
        self,
        message: str,
        agent: dict = None,
        activity: str = None,
        data: Any = None,
        tool_call: Any = None,
        llm_input: Any = None,
        llm_output: Any = None,
    ):
        entry = {
            "message": message,
            "activity": activity,
        }
        if agent:
            entry["agent"] = agent
        if data is not None:
            entry["data"] = data
        if tool_call is not None:
            entry["tool_call"] = tool_call
        if llm_input is not None:
            entry["llm_input"] = llm_input
        if llm_output is not None:
            entry["llm_output"] = llm_output
        self.trace.append(entry)

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.trace, f, indent=2)


class UserGeneratorAgent:
    """
    {desc}
    """.format(
        desc=AGENTS_UPDATE["UserGeneratorAgent"]["description"]
    )

    def __init__(
        self,
        segment: Dict[str, Any],
        agent_config: Dict[str, Any],
        agent_state: Dict[str, int],
        user_id_field: str,
    ):
        self.segment = segment
        self.config = agent_config
        self.state = agent_state
        self.user_id_field = user_id_field
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["UserGeneratorAgent"]["description"]
        self.system_message = AGENTS_UPDATE["UserGeneratorAgent"]["system_message"]
        self.name = "UserGeneratorAgent"
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.llm_config = {
            "api_key": openai_api_key,
            "model": self.model,
        }

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    def generate_user(self, tracer: TracedGroupChat = None) -> SyntheticUser:
        """
        Generate a synthetic user for the segment using OpenAI LLM, assigning a unique user_id.
        Returns a SyntheticUser Pydantic model instance.
        """
        user_id = get_and_increment_user_id(self.state, self.user_id_field)
        prompt = (
            f"Generate a synthetic user profile as a JSON object for the following market segment. "
            f"The JSON must match the provided schema and include all required fields. "
            f"Segment name: {self.segment['nome']}\n"
            f"Segment description: {self.segment['descricao']}\n"
            f"Segment attributes: {json.dumps(self.segment['atributos'], ensure_ascii=False)}\n"
        )
        llm_input = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        try:
            response = openai.ChatCompletion.create(**llm_input)
            llm_output = response["choices"][0]["message"]["content"]
            user_dict = json.loads(llm_output)
            user_dict[self.user_id_field] = str(user_id)
            user = SyntheticUser.model_validate(user_dict)
        except (Exception, PydanticValidationError) as e:
            llm_output = str(e)
            user = None
        if tracer:
            tracer.log(
                message="Generated synthetic user",
                agent=self.get_metadata(),
                activity="generate_user",
                data=user.model_dump() if user else None,
                llm_input=llm_input,
                llm_output=llm_output,
            )
        if user is None:
            raise RuntimeError(f"Failed to generate valid synthetic user: {llm_output}")
        return user


class ValidatorAgent:
    """
    {desc}
    """.format(
        desc=AGENTS_UPDATE["ValidatorAgent"]["description"]
    )

    def __init__(self, schema: Dict[str, Any], agent_config: Dict[str, Any]):
        self.schema = schema
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["ValidatorAgent"]["description"]
        self.system_message = AGENTS_UPDATE["ValidatorAgent"]["system_message"]
        self.name = "ValidatorAgent"
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.llm_config = {
            "api_key": openai_api_key,
            "model": self.model,
        }

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    def validate_user(
        self, user: SyntheticUser, tracer: TracedGroupChat = None
    ) -> CriticOutput:
        """
        Validate a synthetic user using the critic schema and return a CriticOutput Pydantic model.
        """
        # For demonstration, always return a valid output. Replace with real logic as needed.
        output = CriticOutput(score=1.0, issues=[], recommendation="accept")
        if tracer:
            tracer.log(
                message="Validated synthetic user",
                agent=self.get_metadata(),
                activity="validate_user",
                data={"user": user.model_dump(), "critic_output": output.model_dump()},
            )
        return output


class ReviewerAgent:
    """
    {desc}
    """.format(
        desc=AGENTS_UPDATE["ReviewerAgent"]["description"]
    )

    def __init__(self, agent_config: Dict[str, Any]):
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["ReviewerAgent"]["description"]
        self.system_message = AGENTS_UPDATE["ReviewerAgent"]["system_message"]
        self.name = "ReviewerAgent"
        self.model = self.config.get("model", "gpt-3.5-turbo")
        self.llm_config = {
            "api_key": openai_api_key,
            "model": self.model,
        }

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    def review_user(
        self,
        user: SyntheticUser,
        critic_output: CriticOutput,
        tracer: TracedGroupChat = None,
    ) -> dict:
        """
        Review a synthetic user and return a dict with an update_synthetic_user field using the SyntheticUser model.
        """
        # For demonstration, simply return the user as-is in the update_synthetic_user field.
        reviewed = {"update_synthetic_user": user}
        if tracer:
            tracer.log(
                message="Reviewed synthetic user",
                agent=self.get_metadata(),
                activity="review_user",
                data=reviewed,
            )
        return reviewed


class Orchestrator:
    """
    Orchestrates the synthetic user generation workflow, coordinating all agents.
    """

    def __init__(self, config_path: str, agent_config_path: str, agent_state_path: str):
        self.config = load_json(config_path)
        self.agent_config = load_json(agent_config_path)
        self.agent_state = load_json(agent_state_path)
        # Use 'user_id' as default if not defined in config
        self.user_id_field = self.agent_config.get("user_id_field", "user_id")
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
        all_users = []  # List to collect all generated users
        for segment in self.segments:
            self.tracer.log(
                message=f"Processing segment: {segment['nome']}",
                agent=None,
                activity="process_segment",
                data={"segment": segment},
            )
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
                # Generate a user and log all agent context and LLM input/output
                user = generator.generate_user(tracer=self.tracer)
                # Validate the user and log all agent context
                valid, error = validator.validate_user(user, tracer=self.tracer)
                if not valid:
                    self.tracer.log(
                        message=f"Validation failed for user {i+1}",
                        agent=validator.get_metadata(),
                        activity="validation_failed",
                        data={"user": user, "error": error},
                    )
                    # Review and correct the user, log all agent context
                    user = reviewer.review_user(user, error, tracer=self.tracer)
                else:
                    self.tracer.log(
                        message=f"User {i+1} validated successfully",
                        agent=validator.get_metadata(),
                        activity="validation_success",
                        data={"user": user},
                    )
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
