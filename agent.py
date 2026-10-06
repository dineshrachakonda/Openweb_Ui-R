"""
agent.py - A simple LangGraph agent with 5 tools.

Flow:  START -> agent -> (needs a tool?) -> tools -> agent -> ... -> END
"""
from datetime import datetime
from pathlib import Path

from ddgs import DDGS
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

NOTES_DIR = Path("notes")

# Create the notes folder automatically if it does not already exist.
NOTES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------- TOOLS ----
@tool
def web_search(query: str) -> str:
    """Search the internet for up-to-date information on a topic."""
    try:
        results = DDGS().text(query, max_results=4)

        # Return the most useful parts of each search result to the agent.
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results) or "No results."
    except Exception as e:
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Calculate a math expression, e.g. '(120 * 3) / 4'."""
    
    # Restrict the calculator to basic mathematical characters.
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "Only numbers and + - * / ( ) % are allowed."

    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")


@tool
def save_note(title: str, content: str) -> str:
    """Save a note to a text file. Use this to store summaries or results."""
    
    # Turn the title into a safe filename before creating the note.
    name = "".join(c if c.isalnum() else "_" for c in title)[:50] + ".txt"
    (NOTES_DIR / name).write_text(content, encoding="utf-8")
    return f"Saved note as notes/{name}"


@tool
def list_notes() -> str:
    """List all saved notes."""
    
    # Find every text file inside the notes folder.
    files = [f.name for f in NOTES_DIR.glob("*.txt")]
    return "\n".join(files) if files else "No notes saved yet."


# These are the tools that the LLM is allowed to call.
tools = [web_search, calculator, get_current_time, save_note, list_notes]


# ---------------------------------------------------------------- MODEL ----
# Use the local Ollama model and give it access to all five tools.
llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

SYSTEM_PROMPT = (
    "You are a helpful assistant that completes tasks step by step. "
    "Use tools when needed: search for facts, calculate numbers, save notes. "
    "Do not guess facts - search first. When finished, give a short final answer."
)


# ---------------------------------------------------------------- GRAPH ----
def agent_node(state: MessagesState):
    """The 'brain': the LLM looks at the conversation and decides what to do."""
    
    # Give the model the instructions first, followed by the conversation.
    reply = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages": [reply]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)

# ToolNode executes the tool selected by the LLM.
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")

# Decide whether the agent should call a tool or finish the conversation.
builder.add_conditional_edges("agent", tools_condition)

# Send the tool result back to the agent so it can continue reasoning.
builder.add_edge("tools", "agent")

graph = builder.compile()


# Run a simple test directly from this file instead of using Open WebUI.
if __name__ == "__main__":  # quick test without Open WebUI
    q = "Search the latest news about trains in India, summarize in 3 lines and save it as a note."
    for step in graph.stream({"messages": [("user", q)]}, {"recursion_limit": 15}):
        print(step, "\n")