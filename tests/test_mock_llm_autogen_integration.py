"""Tests for verifying the integration between mock LLM and autogen agents."""

import pytest
import autogen
from typing import Dict, Any
from .mock_llm import MockLLM, DynamicMockLLM


class TestAgentIntegration:
    """Tests for agent integration with mock LLM."""
    
    @pytest.fixture
    def agent_response_map(self) -> Dict[str, str]:
        """Fixture providing agent-specific response patterns."""
        return {
            "hello": "Hello! I am an autogen agent.",
            "task": "I will help you with this task.",
            "error": "I encountered an error while processing your request.",
            "success": "I have successfully completed the task.",
            "default": "I am processing your request."
        }
    
    @pytest.fixture
    def mock_llm_agent(self, agent_response_map: Dict[str, str]) -> MockLLM:
        """Fixture providing a MockLLM instance configured for agent testing."""
        return MockLLM(agent_response_map)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_llm_agent):
        """Test agent initialization with mock LLM."""
        # Create an autogen agent with mock LLM
        agent = autogen.AssistantAgent(
            name="test_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=mock_llm_agent
        )
        
        # Verify agent is properly initialized
        assert agent.name == "test_llm_agent"
        assert agent.llm is not None
    
    @pytest.mark.asyncio
    async def test_agent_communication(self, mock_llm_agent):
        """Test agent communication using mock LLM."""
        # Create two agents
        agent1 = autogen.AssistantAgent(
            name="agent1",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=mock_llm_agent
        )
        
        agent2 = autogen.AssistantAgent(
            name="agent2",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=mock_llm_agent
        )
        
        # Test communication between agents
        response = await agent1.generate_reply(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert "Hello! I am an autogen agent" in response
        
        # Verify message history
        history = mock_llm_agent.get_call_history()
        assert len(history) > 0
        assert history[0]["prompt"] == "hello"
    
    @pytest.mark.asyncio
    async def test_agent_task_handling(self, mock_llm_agent):
        """Test agent task handling with mock LLM."""
        agent = autogen.AssistantAgent(
            name="task_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=mock_llm_agent
        )
        
        # Test task processing
        response = await agent.generate_reply(
            messages=[{"role": "user", "content": "task"}]
        )
        assert "I will help you with this task" in response
        
        # Test error handling
        response = await agent.generate_reply(
            messages=[{"role": "user", "content": "error"}]
        )
        assert "I encountered an error" in response
    
    @pytest.mark.asyncio
    async def test_agent_concurrent_operations(self, mock_llm_agent):
        """Test concurrent operations with multiple agents."""
        import asyncio
        
        # Create multiple agents
        agents = [
            autogen.AssistantAgent(
                name=f"agent_{i}",
                llm_config={"config_list": [{"model": "mock_llm"}]},
                llm=mock_llm_agent
            )
            for i in range(3)
        ]
        
        # Test concurrent operations
        async def run_agent(agent):
            return await agent.generate_reply(
                messages=[{"role": "user", "content": "hello"}]
            )
        
        tasks = [run_agent(agent) for agent in agents]
        responses = await asyncio.gather(*tasks)
        
        # Verify responses
        assert len(responses) == 3
        assert all("Hello! I am an autogen agent" in resp for resp in responses)
        
        # Verify history
        history = mock_llm_agent.get_call_history()
        assert len(history) == 3


class TestAgentErrorHandling:
    """Tests for agent error handling with mock LLM."""
    
    @pytest.fixture
    def error_response_map(self) -> Dict[str, str]:
        """Fixture providing error response patterns."""
        return {
            "timeout": "Error: Operation timed out",
            "invalid": "Error: Invalid input",
            "permission": "Error: Permission denied"
        }
    
    @pytest.fixture
    def error_mock_llm_agent(self, error_response_map: Dict[str, str]) -> MockLLM:
        """Fixture providing a MockLLM instance configured for error testing."""
        return MockLLM(error_response_map)
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, error_mock_llm_agent):
        """Test agent timeout handling."""
        agent = autogen.AssistantAgent(
            name="timeout_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=error_mock_llm_agent
        )
        
        response = await agent.generate_reply(
            messages=[{"role": "user", "content": "timeout"}]
        )
        assert "Operation timed out" in response
    
    @pytest.mark.asyncio
    async def test_agent_invalid_input_handling(self, error_mock_llm_agent):
        """Test agent invalid input handling."""
        agent = autogen.AssistantAgent(
            name="invalid_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=error_mock_llm_agent
        )
        
        response = await agent.generate_reply(
            messages=[{"role": "user", "content": "invalid"}]
        )
        assert "Invalid input" in response


class TestAgentDynamicResponses:
    """Tests for agent dynamic responses with mock LLM."""
    
    @pytest.fixture
    def dynamic_response_map(self) -> Dict[str, str]:
        """Fixture providing dynamic response patterns."""
        return {
            "count": "This is response number {count}",
            "time": "Current time is {timestamp}",
            "task": "Task {task_id} completed in {count} steps"
        }
    
    @pytest.fixture
    def dynamic_mock_llm_agent(self, dynamic_response_map: Dict[str, str]) -> DynamicMockLLM:
        """Fixture providing a DynamicMockLLM instance for agent testing."""
        return DynamicMockLLM(dynamic_response_map)
    
    @pytest.mark.asyncio
    async def test_agent_dynamic_counting(self, dynamic_mock_llm_agent):
        """Test agent dynamic counting responses."""
        agent = autogen.AssistantAgent(
            name="count_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=dynamic_mock_llm_agent
        )
        
        # Test multiple responses
        for i in range(3):
            response = await agent.generate_reply(
                messages=[{"role": "user", "content": "count"}]
            )
            assert f"response number {i+1}" in response
    
    @pytest.mark.asyncio
    async def test_agent_dynamic_timing(self, dynamic_mock_llm_agent):
        """Test agent dynamic timing responses."""
        agent = autogen.AssistantAgent(
            name="time_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=dynamic_mock_llm_agent
        )
        
        response = await agent.generate_reply(
            messages=[{"role": "user", "content": "time"}]
        )
        assert "Current time is" in response 


class TestAgentCollaboration:
    """Tests for agent collaboration scenarios."""
    
    @pytest.fixture
    def collaboration_response_map(self) -> Dict[str, str]:
        """Fixture providing collaboration response patterns."""
        return {
            "plan": "I will help plan the next steps.",
            "execute": "I will execute the task.",
            "review": "I will review the results.",
            "approve": "I approve the changes.",
            "reject": "I reject the changes.",
            "suggest": "I suggest the following improvements:",
            "coordinate": "I will coordinate with other agents.",
            "default": "I am ready to collaborate."
        }
    
    @pytest.fixture
    def collaboration_mock_llm(self, collaboration_response_map: Dict[str, str]) -> MockLLM:
        """Fixture providing a MockLLM instance for collaboration testing."""
        return MockLLM(collaboration_response_map)
    
    @pytest.mark.asyncio
    async def test_agent_workflow(self, collaboration_mock_llm):
        """Test a complete agent workflow with multiple agents."""
        # Create specialized agents
        planner = autogen.AssistantAgent(
            name="planner",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=collaboration_mock_llm
        )
        
        executor = autogen.AssistantAgent(
            name="executor",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=collaboration_mock_llm
        )
        
        reviewer = autogen.AssistantAgent(
            name="reviewer",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=collaboration_mock_llm
        )
        
        # Simulate a workflow
        # 1. Planner creates a plan
        plan_response = await planner.generate_reply(
            messages=[{"role": "user", "content": "plan"}]
        )
        assert "plan the next steps" in plan_response
        
        # 2. Executor implements the plan
        execute_response = await executor.generate_reply(
            messages=[{"role": "user", "content": "execute"}]
        )
        assert "execute the task" in execute_response
        
        # 3. Reviewer reviews the results
        review_response = await reviewer.generate_reply(
            messages=[{"role": "user", "content": "review"}]
        )
        assert "review the results" in review_response
        
        # Verify collaboration history
        history = collaboration_mock_llm.get_call_history()
        assert len(history) == 3
        assert history[0]["prompt"] == "plan"
        assert history[1]["prompt"] == "execute"
        assert history[2]["prompt"] == "review"
    
    @pytest.mark.asyncio
    async def test_agent_consensus(self, collaboration_mock_llm):
        """Test agent consensus building."""
        # Create multiple agents for consensus
        agents = [
            autogen.AssistantAgent(
                name=f"agent_{i}",
                llm_config={"config_list": [{"model": "mock_llm"}]},
                llm=collaboration_mock_llm
            )
            for i in range(3)
        ]
        
        # Simulate a consensus building process
        responses = []
        for agent in agents:
            response = await agent.generate_reply(
                messages=[{"role": "user", "content": "suggest"}]
            )
            responses.append(response)
        
        # Verify all agents provided suggestions
        assert len(responses) == 3
        assert all("suggest the following improvements" in resp for resp in responses)
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self, collaboration_mock_llm):
        """Test agent coordination."""
        coordinator = autogen.AssistantAgent(
            name="coordinator",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=collaboration_mock_llm
        )
        
        # Test coordination
        response = await coordinator.generate_reply(
            messages=[{"role": "user", "content": "coordinate"}]
        )
        assert "coordinate with other agents" in response


class TestAgentStatePersistence:
    """Tests for agent state persistence."""
    
    @pytest.fixture
    def state_response_map(self) -> Dict[str, str]:
        """Fixture providing state-related response patterns."""
        return {
            "save": "State saved successfully.",
            "load": "State loaded successfully.",
            "update": "State updated successfully.",
            "clear": "State cleared successfully.",
            "default": "Current state maintained."
        }
    
    @pytest.fixture
    def state_mock_llm(self, state_response_map: Dict[str, str]) -> MockLLM:
        """Fixture providing a MockLLM instance for state testing."""
        return MockLLM(state_response_map)
    
    @pytest.mark.asyncio
    async def test_agent_state_save_load(self, state_mock_llm):
        """Test agent state save and load operations."""
        agent = autogen.AssistantAgent(
            name="state_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=state_mock_llm
        )
        
        # Test state saving
        save_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "save"}]
        )
        assert "State saved successfully" in save_response
        
        # Test state loading
        load_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "load"}]
        )
        assert "State loaded successfully" in load_response
        
        # Verify state operations in history
        history = state_mock_llm.get_call_history()
        assert len(history) == 2
        assert history[0]["prompt"] == "save"
        assert history[1]["prompt"] == "load"
    
    @pytest.mark.asyncio
    async def test_agent_state_update(self, state_mock_llm):
        """Test agent state update operations."""
        agent = autogen.AssistantAgent(
            name="update_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=state_mock_llm
        )
        
        # Test state update
        update_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "update"}]
        )
        assert "State updated successfully" in update_response
        
        # Verify state is maintained
        default_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "default"}]
        )
        assert "Current state maintained" in default_response
    
    @pytest.mark.asyncio
    async def test_agent_state_clear(self, state_mock_llm):
        """Test agent state clear operation."""
        agent = autogen.AssistantAgent(
            name="clear_agent",
            llm_config={"config_list": [{"model": "mock_llm"}]},
            llm=state_mock_llm
        )
        
        # Test state clearing
        clear_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "clear"}]
        )
        assert "State cleared successfully" in clear_response
        
        # Verify state is cleared
        default_response = await agent.generate_reply(
            messages=[{"role": "user", "content": "default"}]
        )
        assert "Current state maintained" in default_response 