from typing import Dict, Any
from .base_supervisor import BaseSupervisor


class UpdateManifestSupervisor(BaseSupervisor):
    """Supervisor class for the update_manifest program."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the update manifest supervisor.

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
        elif name == "manifest_updater":
            from .manifest_updater_agent import ManifestUpdaterAgent

            return ManifestUpdaterAgent(config)
        elif name == "logging_config":
            from .logging_config_agent import LoggingConfigAgent

            return LoggingConfigAgent(config)
        elif name == "validation":
            from .validation_agent import ValidationAgent

            return ValidationAgent(config)
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

        # Add manifest-specific context
        if task.get("type") == "manifest_update":
            base_message += f"\nManifest Version: {task.get('metadata', {}).get('version', 'unknown')}"
            base_message += f"\nUpdate Type: {task.get('metadata', {}).get('update_type', 'unknown')}"

        return base_message
