import json
from typing import Dict, Any, Optional


class WorkflowCriticAgent:
    """
    Analyzes a workflow trace and human feedback to generate a critique and action plan for improvement.
    Optionally writes the output to a file if output_file is provided.
    """

    def __init__(self, trace_path: str, output_file: Optional[str] = None):
        self.trace_path = trace_path
        self.output_file = output_file
        self.trace = self._load_trace()
        self.agent_metadata = self.trace.get("agents", {})
        self.actions = self.trace.get("actions", [])
        self.group_description = self.trace.get("group_description", "")

    def _load_trace(self) -> Dict[str, Any]:
        with open(self.trace_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def critique(self, human_feedback: str) -> Dict[str, Any]:
        # Simple rule-based analysis for demonstration
        critique_lines = []
        action_plan = []
        critique_lines.append(f"Group description: {self.group_description}")
        critique_lines.append(f"Human feedback: {human_feedback}")
        critique_lines.append(f"Total actions: {len(self.actions)}")
        # Example: check for agents with no actions
        agent_action_counts = {name: 0 for name in self.agent_metadata}
        for action in self.actions:
            agent = action.get("agent")
            if agent in agent_action_counts:
                agent_action_counts[agent] += 1
        for agent, meta in self.agent_metadata.items():
            if agent_action_counts[agent] == 0:
                critique_lines.append(
                    f"Agent '{agent}' ({meta.get('description','')}) did not perform any actions."
                )
                action_plan.append(
                    f"Review the role and integration of agent '{agent}'."
                )
        # Example: suggest reviewing system messages
        for agent, meta in self.agent_metadata.items():
            if not meta.get("system_message"):
                action_plan.append(
                    f"Add or clarify system_message for agent '{agent}'."
                )
        # Example: generic improvement step
        action_plan.append("Consider using an LLM for deeper workflow analysis.")
        result = {"critique": "\n".join(critique_lines), "action_plan": action_plan}
        if self.output_file:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        return result
