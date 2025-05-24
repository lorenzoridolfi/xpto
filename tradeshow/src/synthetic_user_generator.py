import json
import os
from typing import Any, Dict, Optional
from pydantic import ValidationError as PydanticValidationError
from autogen_extensions.log_utils import get_logger
from autogen_extensions.json_validation import validate_json
from tradeshow.src.pydantic_schema import (
    SyntheticUserDraft,
    SyntheticUserReviewed,
    CriticOutput,
)
import gc
import time
import argparse
from autogen_extensions.tracing import TracingMixin
from autogen_extensions.config_utils import ROOT_FOLDER
from autogen_extensions.llm_base_agent import LLMBaseAgent
import asyncio

# --- Logging Configuration ---
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
LOG_FILE = os.path.join(LOG_DIR, "synthetic_user_generator.log")
os.makedirs(LOG_DIR, exist_ok=True)
logger = get_logger("synthetic_user_generator")

# --- OpenAI Client Setup ---
# Remove: openai_client = get_openai_client()

PROJECT_ROOT = os.environ.get(
    "AUTOGEN_ROOT_FOLDER", "/Users/lorenzo/Sync/Source/AI/autogen"
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.json")
AGENT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config_agents.json")
AGENT_STATE_PATH = os.path.join(
    os.path.dirname(__file__), "../config_agents_state.json"
)
SEGMENTS_PATH = os.path.join(os.path.dirname(__file__), "../input/segments.json")
SEGMENT_SCHEMA_PATH = os.path.join(
    ROOT_FOLDER, "tradeshow", "schema", "segments_schema.json"
)
AGENTS_UPDATE_PATH = os.path.join(
    os.path.dirname(__file__), "../other/agents_update.json"
)

# --- Load agent updates ---
with open(AGENTS_UPDATE_PATH) as f:
    AGENTS_UPDATE = json.load(f)


def load_json(path: str) -> Any:
    logger.debug(f"Loading JSON file: {path}")
    with open(path) as f:
        return json.load(f)


def save_json(path: str, data: Any):
    logger.debug(f"Saving JSON to: {path}")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_and_increment_user_id(state: Dict[str, int], user_id_field: str) -> int:
    number = state[user_id_field]
    logger.debug(f"Current user_id: {number} (field: {user_id_field})")
    state[user_id_field] += 1
    save_json(AGENT_STATE_PATH, state)
    logger.info(f"Incremented user_id to {state[user_id_field]}")
    return number


def validate_segments_schema():
    logger.info("Validating segments.json against schema...")
    schema = load_json(SEGMENT_SCHEMA_PATH)
    segments = load_json(SEGMENTS_PATH)
    try:
        validate_json(segments, schema)
        logger.info("segments.json validation successful.")
    except Exception as e:
        logger.error(f"segments.json validation error: {e}")
        raise RuntimeError(f"segments.json validation error: {e}")


# Validate segments.json at import/run time
validate_segments_schema()


class TracedGroupChat(TracingMixin):
    """
    Class responsible for logging all actions and messages for traceability.
    Each log entry includes agent metadata and activity context.
    """

    def __init__(self, log_path: str):
        super().__init__(trace_path=log_path)
        logger.info(f"Trace log initialized at {log_path}")

    def log(
        self,
        message: str,
        agent: dict = None,
        activity: str = None,
        data: Any = None,
        tool_call: Any = None,
        llm_input: Any = None,
        llm_output: Any = None,
        duration_seconds: float = None,
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
        if duration_seconds is not None:
            entry["duration_seconds"] = duration_seconds
        self.log_event("trace", entry)

    def save(self):
        self.save_trace()


class UserGeneratorAgent(LLMBaseAgent):
    """
    Generates a SyntheticUserDraft (before review) for a given segment.
    Optionally logs events using a tracer (any object with a .log method).
    """

    def __init__(
        self,
        segment,
        agent_config,
        agent_state,
        user_id_field,
        schema,
        tracer=None,
        llm_call=None,
    ):
        super().__init__(llm_call=llm_call)
        self.segment = segment
        self.config = agent_config
        self.state = agent_state
        self.user_id_field = user_id_field
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["UserGeneratorAgent"]["description"]
        self.system_message = AGENTS_UPDATE["UserGeneratorAgent"]["system_message"]
        self.name = "UserGeneratorAgent"
        self.model = self.config.get("model", "gpt-4o")
        self.schema = schema
        logger.info(
            f"Initializing UserGeneratorAgent for segment: {segment.get('nome')}"
        )
        self.tracer = tracer
        logger.debug(f"UserGeneratorAgent initialized: {self.get_metadata()}")

    def get_metadata(self) -> dict:
        logger.debug("UserGeneratorAgent.get_metadata called")
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    async def generate_user(self) -> SyntheticUserDraft:
        logger.debug(
            f"UserGeneratorAgent.generate_user called for segment: {self.segment.get('nome')}"
        )
        user_id = get_and_increment_user_id(self.state, self.user_id_field)
        logger.debug(
            f"Generating user for segment: {self.segment.get('nome')} with user_id: {user_id}"
        )
        segment_json = json.dumps(self.segment, ensure_ascii=False, indent=2)
        schema_json = json.dumps(self.schema, ensure_ascii=False, indent=2)
        prompt = (
            f"{self.description}\n"
            f"Segment: {segment_json}\n"
            f"Here is the required JSON schema for the output:\n{schema_json}\n"
            "Respond ONLY with a valid JSON object matching the schema above. Do not include any extra text or explanation."
        )
        logger.debug(f"User prompt constructed (full segment): {prompt}")
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        logger.debug(f"Calling LLM with messages: {messages}")
        start_time = time.time()
        try:
            response = await self._call_llm(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start_time
            logger.info(f"LLM user generation call took {elapsed:.2f} seconds.")
            user_json = response.choices[0].message.content
            user = SyntheticUserDraft.model_validate_json(user_json)
            user.id_usuario = str(user_id)
            logger.info(
                f"Generated id_usuario {user.id_usuario} for segment {self.segment.get('nome')}"
            )
            logger.debug(f"LLM response: {user}")
            if self.tracer:
                self.tracer.log(
                    message="LLM user generation call",
                    agent=self.get_metadata(),
                    activity="generate_user",
                    data=user.model_dump(),
                    llm_input=messages,
                    llm_output=user_json,
                    duration_seconds=elapsed,
                )
        except PydanticValidationError as e:
            logger.error(f"Pydantic validation error in UserGeneratorAgent: {e}")
            if self.tracer:
                self.tracer.log(
                    message="Pydantic validation error in UserGeneratorAgent",
                    agent=self.get_metadata(),
                    activity="generate_user",
                    data=None,
                    llm_input=messages,
                    llm_output=str(e),
                )
            raise RuntimeError(f"Failed to generate valid synthetic user: {e}")
        if self.tracer:
            self.tracer.log(
                message="Generated synthetic user",
                agent=self.get_metadata(),
                activity="generate_user",
                data=user.model_dump(),
                llm_input=messages,
                llm_output=user.model_dump(),
            )
        logger.debug(f"UserGeneratorAgent.generate_user returning user: {user}")
        return user


class ValidatorAgent(LLMBaseAgent):
    """
    Validates a SyntheticUserDraft and produces a CriticOutput. Optionally logs events using a tracer (any object with a .log method).
    """

    def __init__(self, schema, agent_config, tracer=None, llm_call=None):
        super().__init__(llm_call=llm_call)
        self.schema = schema
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["ValidatorAgent"]["description"]
        self.system_message = AGENTS_UPDATE["ValidatorAgent"]["system_message"]
        self.name = "ValidatorAgent"
        self.model = self.config.get("model", "gpt-4o")
        self.tracer = tracer
        logger.debug(f"ValidatorAgent initialized: {self.get_metadata()}")

    def get_metadata(self) -> dict:
        logger.debug("ValidatorAgent.get_metadata called")
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    async def validate_user(self, user: SyntheticUserDraft) -> CriticOutput:
        logger.debug(f"ValidatorAgent.validate_user called for user: {user}")
        schema_json = json.dumps(self.schema, ensure_ascii=False, indent=2)
        prompt = (
            f"{self.description}\n"
            f"User: {user.model_dump_json()}\n"
            f"Here is the required JSON schema for the output:\n{schema_json}\n"
            "Respond ONLY with a valid JSON object matching the schema above. Do not include any extra text or explanation."
        )
        logger.debug(f"Validator prompt constructed: {prompt}")
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        logger.debug(f"Calling LLM with messages: {messages}")
        start_time = time.time()
        try:
            response = await self._call_llm(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start_time
            logger.info(f"LLM validation call took {elapsed:.2f} seconds.")
            output_json = response.choices[0].message.content
            output = CriticOutput.model_validate_json(output_json)
            logger.info(f"Validation result: {output}")
            if self.tracer:
                self.tracer.log(
                    message="LLM validation call",
                    agent=self.get_metadata(),
                    activity="validate_user",
                    data={
                        "user": user.model_dump(),
                        "critic_output": output.model_dump(),
                    },
                    llm_input=messages,
                    llm_output=output_json,
                    duration_seconds=elapsed,
                )
        except PydanticValidationError as e:
            logger.error(f"Pydantic validation error in ValidatorAgent: {e}")
            if self.tracer:
                self.tracer.log(
                    message="Pydantic validation error in ValidatorAgent",
                    agent=self.get_metadata(),
                    activity="validate_user",
                    data=None,
                    llm_input=messages,
                    llm_output=str(e),
                )
            raise RuntimeError(f"Failed to validate synthetic user: {e}")
        if self.tracer:
            self.tracer.log(
                message="Validated synthetic user",
                agent=self.get_metadata(),
                activity="validate_user",
                data={"user": user.model_dump(), "critic_output": output.model_dump()},
                llm_input=messages,
                llm_output=output.model_dump(),
            )
        logger.debug(f"ValidatorAgent.validate_user returning output: {output}")
        return output


class ReviewerAgent(LLMBaseAgent):
    """
    Reviews a SyntheticUserDraft and CriticOutput, and returns a SyntheticUserReviewed. Optionally logs events using a tracer (any object with a .log method).
    """

    def __init__(self, agent_config, schema, tracer=None, llm_call=None):
        super().__init__(llm_call=llm_call)
        self.config = agent_config
        self.temperature = self.config["temperature"]
        self.description = AGENTS_UPDATE["ReviewerAgent"]["description"]
        self.system_message = AGENTS_UPDATE["ReviewerAgent"]["system_message"]
        self.name = "ReviewerAgent"
        self.model = self.config.get("model", "gpt-4o")
        self.schema = schema
        self.tracer = tracer
        logger.debug(f"ReviewerAgent initialized: {self.get_metadata()}")

    def get_metadata(self) -> dict:
        logger.debug("ReviewerAgent.get_metadata called")
        return {
            "name": self.name,
            "description": self.description,
            "system_message": self.system_message,
            "temperature": self.temperature,
            "model": self.model,
        }

    async def review_user(
        self, user: SyntheticUserDraft, critic_output: CriticOutput
    ) -> dict:
        logger.debug(
            f"ReviewerAgent.review_user called for user: {user}, critic_output: {critic_output}"
        )
        schema_json = json.dumps(self.schema, ensure_ascii=False, indent=2)
        prompt = (
            f"{self.description}\n"
            f"User: {user.model_dump_json()}\n"
            f"Critic: {critic_output.model_dump_json()}\n"
            f"Here is the required JSON schema for the output:\n{schema_json}\n"
            "Respond ONLY with a valid JSON object matching the schema above. Do not include any extra text or explanation."
        )
        logger.debug(f"Reviewer prompt constructed: {prompt}")
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]
        logger.debug(f"Calling LLM with messages: {messages}")
        start_time = time.time()
        try:
            response = await self._call_llm(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
            )
            elapsed = time.time() - start_time
            logger.info(f"LLM review call took {elapsed:.2f} seconds.")
            improved_user_json = response.choices[0].message.content
            improved_user = SyntheticUserReviewed.model_validate_json(
                improved_user_json
            )
            if not improved_user.avaliacao or not improved_user.avaliacao.critica:
                improved_user.avaliacao.critica = str(critic_output)
            if not improved_user.avaliacao.revisao:
                improved_user.avaliacao.revisao = (
                    f"Reviewed after critic: {critic_output}"
                )
            logger.info(f"Review result: {improved_user}")
            if self.tracer:
                self.tracer.log(
                    message="LLM review call",
                    agent=self.get_metadata(),
                    activity="review_user",
                    data=improved_user.model_dump(),
                    llm_input=messages,
                    llm_output=improved_user_json,
                    duration_seconds=elapsed,
                )
        except PydanticValidationError as e:
            logger.error(f"Pydantic validation error in ReviewerAgent: {e}")
            if self.tracer:
                self.tracer.log(
                    message="Pydantic validation error in ReviewerAgent",
                    agent=self.get_metadata(),
                    activity="review_user",
                    data=None,
                    llm_input=messages,
                    llm_output=str(e),
                )
            raise RuntimeError(f"Failed to review synthetic user: {e}")
        reviewed = {"update_synthetic_user": improved_user}
        if self.tracer:
            self.tracer.log(
                message="Reviewed synthetic user",
                agent=self.get_metadata(),
                activity="review_user",
                data={"update_synthetic_user": improved_user.model_dump()},
                llm_input=messages,
                llm_output=improved_user.model_dump(),
            )
        logger.debug(f"ReviewerAgent.review_user returning reviewed: {reviewed}")
        return reviewed


def resolve_path(base_dir, path):
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


class Orchestrator:
    """
    Orchestrates the synthetic user generation workflow, coordinating all agents and passing a tracer for logging.
    """

    def __init__(
        self,
        config_path: str,
        agent_config_path: str,
        agent_state_path: str,
        output_file: str,
        append: bool,
        tracer: Optional[Any] = None,
        llm_call: Optional[Any] = None,
    ):
        logger.debug(
            f"Orchestrator.__init__ called with config_path={config_path}, agent_config_path={agent_config_path}, agent_state_path={agent_state_path}, output_file={output_file}, append={append}"
        )
        self.config = load_json(config_path)
        self.agent_config = load_json(agent_config_path)
        self.agent_state = load_json(agent_state_path)
        self.user_id_field = self.agent_config.get("user_id_field", "user_id")
        logger.info(f"Orchestrator initialized with user_id_field={self.user_id_field}")
        with open(os.path.join(PROJECT_ROOT, self.config["input_segment_file"])) as f:
            self.segments = json.load(f)["segmentos"]
        logger.debug(f"Loaded segments: {self.segments}")
        schema_path = self.config["synthetic_user_schema_file"]
        if not os.path.isabs(schema_path):
            schema_path = os.path.join(PROJECT_ROOT, schema_path)
        with open(schema_path) as f:
            self.schema = json.load(f)
        logger.debug(f"Loaded schema from {schema_path}")
        self.output_file = output_file
        self.append = append
        output_dir = os.path.dirname(self.output_file)
        os.makedirs(output_dir, exist_ok=True)
        self.tracer = tracer or TracedGroupChat(
            os.path.join(PROJECT_ROOT, self.config["log_file"])
        )
        self.llm_call = llm_call
        logger.info(f"Orchestrator output_file set to {self.output_file}")

    async def run(self):
        logger.info("Orchestrator.run started")
        all_users = []
        # Handle append logic
        if self.append and os.path.exists(self.output_file):
            logger.info(f"Appending to existing output file: {self.output_file}")
            with open(self.output_file, "r", encoding="utf-8") as f:
                all_users = json.load(f)
            # Continue user_id numbering from last user
            if all_users:
                last_user_id = max(u.get(self.user_id_field, 0) for u in all_users)
                self.agent_state[self.user_id_field] = last_user_id + 1
                logger.info(f"Continuing user_id from {last_user_id + 1}")
            else:
                self.agent_state[self.user_id_field] = 1
        else:
            # Reset user_id to 1 at the start of each run
            self.agent_state[self.user_id_field] = 1
        save_json(AGENT_STATE_PATH, self.agent_state)
        overall_user_count = len(all_users)
        for seg_idx, segment in enumerate(self.segments, start=1):
            logger.info(f"Processing segment: {segment['nome']}")
            self.tracer.log(
                message=f"Processing segment: {segment['nome']}",
                agent=None,
                activity="process_segment",
                data={"segment": segment},
            )
            num_usuarios = segment.get("num_usuarios")
            if not isinstance(num_usuarios, int) or num_usuarios < 1:
                logger.error(
                    f"Invalid num_usuarios for segment {segment.get('nome', '<unknown>')}: {num_usuarios}"
                )
                raise ValueError(
                    f"Segment '{segment.get('nome', '<unknown>')}' is missing a valid 'num_usuarios' field."
                )
            logger.info(
                f"Generating {num_usuarios} users for segment {segment['nome']}"
            )
            generator = UserGeneratorAgent(
                segment,
                self.agent_config["UserGeneratorAgent"],
                self.agent_state,
                self.user_id_field,
                self.schema,
                tracer=self.tracer,
                llm_call=self.llm_call,
            )
            validator = ValidatorAgent(
                self.schema,
                self.agent_config["ValidatorAgent"],
                tracer=self.tracer,
                llm_call=self.llm_call,
            )
            reviewer = ReviewerAgent(
                self.agent_config["ReviewerAgent"],
                self.schema,
                tracer=self.tracer,
                llm_call=self.llm_call,
            )
            segment_users = []
            for i in range(num_usuarios):
                overall_user_count += 1
                highlight_msg = (
                    f"\n{'='*60}\n"
                    f"  STARTING USER CREATION\n"
                    f"  Segment: {segment['nome']} (Segment {seg_idx}/{len(self.segments)})\n"
                    f"  User in segment: {i+1}/{num_usuarios}\n"
                    f"  User overall: {overall_user_count}\n"
                    f"{'='*60}\n"
                )
                logger.info(highlight_msg)
                logger.debug("Running garbage collection before user generation...")
                gc.collect()
                logger.debug("Garbage collection complete.")
                logger.debug(
                    f"Generating user {i+1}/{num_usuarios} for segment {segment['nome']}"
                )
                user = await generator.generate_user()
                logger.debug(
                    f"Validating user {i+1}/{num_usuarios} for segment {segment['nome']}"
                )
                critic_output = await validator.validate_user(user)
                if critic_output.recommendation != "accept":
                    logger.warning(
                        f"Validation failed for user {i+1} in segment {segment['nome']}"
                    )
                    self.tracer.log(
                        message=f"Validation failed for user {i+1}",
                        agent=validator.get_metadata(),
                        activity="validation_failed",
                        data={
                            "user": user.model_dump(),
                            "critic_output": critic_output.model_dump(),
                        },
                    )
                    logger.debug(
                        f"Reviewing user {i+1}/{num_usuarios} for segment {segment['nome']}"
                    )
                    reviewed = await reviewer.review_user(user, critic_output)
                    user = reviewed["update_synthetic_user"]
                else:
                    logger.info(
                        f"User {i+1} validated successfully for segment {segment['nome']}"
                    )
                    self.tracer.log(
                        message=f"User {i+1} validated successfully",
                        agent=validator.get_metadata(),
                        activity="validation_success",
                        data={"user": user.model_dump()},
                    )
                segment_users.append(user.model_dump())
            all_users.extend(segment_users)
        logger.info(f"Saving all users to {self.output_file}")
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(all_users, f, indent=2, ensure_ascii=False)
        logger.info("Saving trace log")
        self.tracer.save()
        logger.info("Saving updated agent state")
        save_json(AGENT_STATE_PATH, self.agent_state)
        logger.info("Orchestrator.run completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_dir = os.path.join(PROJECT_ROOT, "tradeshow/data")
    default_output = os.path.join(data_dir, "synthetic_users.json")
    parser.add_argument(
        "-o", "--output", default=default_output, help="Output file path"
    )
    parser.add_argument(
        "-a", "--append", action="store_true", help="Append to existing file"
    )
    args = parser.parse_args()

    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(args.output) and not args.append:
        print(
            f"System: Output file '{args.output}' already exists. Use -a or --append to add users."
        )
        exit(1)

    orchestrator = Orchestrator(
        CONFIG_PATH, AGENT_CONFIG_PATH, AGENT_STATE_PATH, args.output, args.append
    )
    asyncio.run(orchestrator.run())
