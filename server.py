"""
server.py - Connects the LangGraph agent to an OpenAI-compatible API.
Open WebUI can communicate with the agent through these endpoints.
"""
import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent import graph

MODEL_NAME = "research-agent"
app = FastAPI()


# Convert messages received from Open WebUI into LangChain message objects.
def to_langchain(messages):
    role_map = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
    out = []
    for m in messages:
        content = m["content"]

        # Some clients send the message content as a list instead of plain text.
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))

        out.append(role_map.get(m["role"], HumanMessage)(content=content))
    return out


# Format the agent's output so Open WebUI can display it as a streamed response.
def chunk(text="", done=False):
    data = {
        "id": "chatcmpl-agent",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {} if done else {"content": text},
            "finish_reason": "stop" if done else None,
        }],
    }
    return f"data: {json.dumps(data)}\n\n"


# Run the LangGraph agent and send its progress to Open WebUI in real time.
def stream_agent(messages):
    cfg = {"recursion_limit": 15}

    # Stream every node update instead of waiting for the entire agent to finish.
    for update in graph.stream({"messages": messages}, cfg, stream_mode="updates"):
        for node, output in update.items():
            for msg in output["messages"]:

                # Show which tool the agent decided to use.
                if node == "agent" and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield chunk(f"🔧 **Using tool:** `{tc['name']}` {tc['args']}\n\n")

                # Send the agent's normal response to the UI.
                elif node == "agent":
                    yield chunk(msg.content)

                # Tell the user when a tool has finished running.
                elif node == "tools":
                    yield chunk(f"✅ **Done:** `{msg.name}`\n\n")

    # Signal that the agent has completely finished.
    yield chunk(done=True)
    yield "data: [DONE]\n\n"


# Open WebUI uses this endpoint to discover the available model.
@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "me"}]}


# Main endpoint used by Open WebUI to send chat messages to the agent.
@app.post("/v1/chat/completions")
def chat(body: dict):
    messages = to_langchain(body["messages"])

    # Use streaming when Open WebUI requests live output.
    if body.get("stream"):
        return StreamingResponse(stream_agent(messages), media_type="text/event-stream")

    # Otherwise, wait for the agent to finish and return the complete response.
    result = graph.invoke({"messages": messages}, {"recursion_limit": 15})
    return {
        "id": "chatcmpl-agent",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result["messages"][-1].content},
            "finish_reason": "stop",
        }],
    }