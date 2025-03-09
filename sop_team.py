import json
import logging
import os
from typing import Any, Awaitable, Callable, Optional

import aiofiles
import yaml
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.messages import TextMessage, UserInputRequestedEvent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from documents.fetch_sop import fetch_document
import faiss
import ollama
import numpy as np
import pickle

logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

state_path = "sop_team_state.json"
history_path = "sop_team_history.json"

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def root():
    """Serve the chat interface HTML file."""
    return FileResponse("sop_team.html")


def search_documents(query: str) -> list:
    # Load the FAISS index from a file
    index = faiss.read_index("embeddings/faiss_index_file.idx")
    model = 'nomic-embed-text'
    query_response = ollama.embeddings(model=model, prompt=query)
    query_embedd = np.array([query_response['embedding']]).astype('float32')
    # Normalize the query embedding
    query_embedd = query_embedd / np.linalg.norm(query_embedd, axis=1, keepdims=True)
    D, I = index.search(query_embedd, k=10)
    distances, indices = D[0], I[0]
    ret = []

    for i in range(len(indices)):
        doc_id = indices[i]
        distance = distances[i]
        doc_data = fetch_document(doc_id)

        ret.append(
            {"doc_is": doc_id,
             "doc_path": doc_data['path'],
             "doc_content": doc_data['content']
             })
    return ret


async def get_team(
        user_input_func: Callable[[str, Optional[CancellationToken]], Awaitable[str]],
) -> RoundRobinGroupChat:
    model_client = OpenAIChatCompletionClient(
        model="llama3.2:latest",
        base_url="http://localhost:11434/v1",
        api_key="placeholder",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks. Don't take multiple turns.
        Your team members are:
            SearchAgent: Searches for information
            SummaryAgent: Performs document summary

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    search_agent = AssistantAgent(
        "SearchAgent",
        description="An agent for searching information from internal storage.",
        tools=[search_documents],
        model_client=model_client,
        # reflect_on_tool_use=True,
        system_message="""
        You are a document search agent.
        Your only tool is search_tool - use it to find document related to user query.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    summarize_agent = AssistantAgent(
        "SummaryAgent",
        description="An agent for summarizing the information fetched by searchAgent",
        model_client=model_client,
        system_message="""
        You are a summary agent. Summarize the finding of the documents.
        You will be called just once.
        """,
    )

    # text_mention_termination = TextMentionTermination("TERMINATE")
    # termination = text_mention_termination | max_messages_termination
    max_messages_termination = MaxMessageTermination(max_messages=3)
    termination = max_messages_termination


    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """

    team = SelectorGroupChat(
        [planning_agent, search_agent, summarize_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=False,  # Allow an agent to speak multiple turns in a row.
    )
    return team


async def get_history() -> list[dict[str, Any]]:
    """Get chat history from file."""
    if not os.path.exists(history_path):
        return []
    async with aiofiles.open(history_path, "r") as file:
        return json.loads(await file.read())


@app.get("/history")
async def history() -> list[dict[str, Any]]:
    try:
        return await get_history()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()

    # User input function used by the team.
    async def _user_input(prompt: str, cancellation_token: CancellationToken | None) -> str:
        data = await websocket.receive_json()
        message = TextMessage.model_validate(data)
        return message.content

    try:
        while True:
            # Get user message.
            data = await websocket.receive_json()
            request = TextMessage.model_validate(data)

            try:
                # Get the team and respond to the message.
                team = await get_team(_user_input)
                history = await get_history()
                stream = team.run_stream(task=request)
                async for message in stream:
                    if isinstance(message, TaskResult):
                        continue
                    await websocket.send_json(message.model_dump())
                    if not isinstance(message, UserInputRequestedEvent):
                        # Don't save user input events to history.
                        history.append(message.model_dump())


            except Exception as e:
                # Send error message to client
                error_message = {
                    "type": "error",
                    "content": f"Error: {str(e)}",
                    "source": "system"
                }
                await websocket.send_json(error_message)
                # Re-enable input after error
                await websocket.send_json({
                    "type": "UserInputRequestedEvent",
                    "content": "An error occurred. Please try again.",
                    "source": "system"
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "content": f"Unexpected error: {str(e)}",
                "source": "system"
            })
        except:
            pass


# Example usage
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
