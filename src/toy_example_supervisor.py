from typing import Dict, Any
from .base_supervisor import BaseSupervisor


class ToyExampleSupervisor(BaseSupervisor):
    """Supervisor class for the toy_example program."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the toy example supervisor.

        Args:
            config: Configuration dictionary containing agent settings
        """
        super().__init__(config)

    async def _create_agent(self, name: str, config: Dict[str, Any]) -> Any:
        """Create an agent instance based on its type and configuration.

        Args:
            name: Name of the agent
            config: Agent configuration

        Returns:
            Initialized agent instance
        """
        if name == "file_reader":
            from .file_reader_agent import FileReaderAgent

            return FileReaderAgent(config)
        elif name == "writer":
            from .writer_agent import WriterAgent

            return WriterAgent(config)
        elif name == "verifier":
            from .verifier_agent import VerifierAgent

            return VerifierAgent(config)
        elif name == "quality":
            from .quality_agent import QualityAgent

            return QualityAgent(config)
        else:
            return await super()._create_agent(name, config)

    def _task_to_message(self, task: Dict[str, Any]) -> str:
        """Convert task dictionary to chat message.

        Args:
            task: Task dictionary to convert

        Returns:
            Formatted chat message
        """
        base_message = super()._task_to_message(task)

        # Add toy example-specific context
        if task.get("type") == "file_processing":
            base_message += (
                f"\nProcessing Mode: {task.get('metadata', {}).get('mode', 'unknown')}"
            )
            base_message += f"\nOutput Format: {task.get('metadata', {}).get('output_format', 'unknown')}"

        return base_message
