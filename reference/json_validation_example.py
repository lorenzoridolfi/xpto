from typing import Literal
from pydantic import BaseModel
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
import asyncio


# The response format for the agent as a Pydantic base model.
class AgentResponse(BaseModel):
    thoughts: str
    response: Literal["happy", "sad", "neutral"]


async def main():
    # Create an agent that uses the OpenAI GPT-4o model with the custom response format.
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        response_format=AgentResponse,  # type: ignore
    )

    # Send a message list to the model and await the response.
    messages = [
        UserMessage(content="I am happy.", source="user"),
    ]
    response = await model_client.create(messages=messages)
    assert isinstance(response.content, str)
    parsed_response = AgentResponse.model_validate_json(response.content)
    print(parsed_response.thoughts)
    print(parsed_response.response)

    # Close the connection to the model client.
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
