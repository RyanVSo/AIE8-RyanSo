"""
Simple Client Agent using LangGraph to communicate with A2A Server

This agent acts as a client that uses the A2A protocol to send requests 
to the existing server agent and processes the responses.
"""
import asyncio
import logging
import os
from typing import Annotated, Dict, Any, List
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# Note: removed pydantic import as A2AClientTool is now a regular class
import re
from typing_extensions import TypedDict

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClientAgentState(TypedDict):
    """State for the client agent that tracks conversation with A2A server."""
    messages: Annotated[List, add_messages]
    server_response: str  # Latest response from server
    is_complete: bool  # Whether the conversation is complete
    connection_ready: bool  # Whether A2A connection is established


# A2A client is now created fresh for each request to avoid serialization issues

def extract_text_from_parts(parts):
    """Extract text content from A2A message parts."""
    if not parts:
        return None
    
    for part in parts:
        if hasattr(part, 'root') and hasattr(part.root, 'text'):
            return part.root.text
    return None


async def initialize_a2a_connection(state: ClientAgentState) -> Dict[str, Any]:
    """Test A2A server connection."""
    logger.info("Testing A2A connection...")
    
    try:
        # Just test that we can reach the server and get the agent card
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url='http://localhost:10000',
            )
            agent_card = await resolver.get_agent_card()
            logger.info(f"Successfully connected to A2A server: {agent_card.name}")
        
        return {
            "connection_ready": True,
            "messages": [AIMessage(content="Connected to A2A server successfully.")]
        }
    except Exception as e:
        error_msg = f"Failed to connect to A2A server: {e}"
        logger.error(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "is_complete": True,
            "connection_ready": False
        }


async def send_to_a2a_server(state: ClientAgentState) -> Dict[str, Any]:
    """Send the user's message to the A2A server and get response."""
    if not state.get("connection_ready"):
        return {
            "messages": [AIMessage(content="A2A connection not ready. Please initialize connection first.")],
            "is_complete": True
        }
    
    # Get the last human message
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not human_messages:
        return {
            "messages": [AIMessage(content="No user message found to send to server.")],
            "is_complete": True
        }
    
    last_human_message = human_messages[-1]
    logger.info(f"Sending message to A2A server: {last_human_message.content}")
    
    try:
        # Create fresh A2A client for this request
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url='http://localhost:10000',
            )
            agent_card = await resolver.get_agent_card()
            client = A2AClient(
                httpx_client=httpx_client, 
                agent_card=agent_card
            )
            
            # Create the message payload
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': last_human_message.content}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            
            request = SendMessageRequest(
                id=str(uuid4()), 
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send message to A2A server
            response = await client.send_message(request)
            
            # Extract the response content from A2A response
            server_content = "No response content found"
            
            if response.root and response.root.result:
                result = response.root.result
                
                # Try to extract from artifacts first (completed tasks)
                if hasattr(result, 'artifacts') and result.artifacts:
                    for artifact in result.artifacts:
                        text_content = extract_text_from_parts(getattr(artifact, 'parts', None))
                        if text_content:
                            server_content = text_content
                            break
                
                # If no artifacts, try to extract from message (input_required, working states)
                if server_content == "No response content found" and hasattr(result, 'message') and result.message:
                    message = result.message
                    text_content = extract_text_from_parts(getattr(message, 'parts', None))
                    if text_content:
                        server_content = text_content
                
                # If still no content, check if there are any messages in the result
                if server_content == "No response content found":
                    # Sometimes the response might be in different structures
                    # Let's try to extract any text content we can find
                    result_str = str(result)
                    if 'text=' in result_str:
                        # Try to extract text using string parsing as fallback
                        text_matches = re.findall(r"text='([^']*)'", result_str)
                        if text_matches:
                            server_content = text_matches[0]
                        else:
                            text_matches = re.findall(r'text="([^"]*)"', result_str)
                            if text_matches:
                                server_content = text_matches[0]
                
                # If we still don't have content, show the status for debugging
                if server_content == "No response content found":
                    status = getattr(result, 'status', 'unknown')
                    server_content = f"Task {status} - response structure: {type(result).__name__}"
                    
                    # Add some debug info about what we found
                    debug_info = []
                    if hasattr(result, 'artifacts'):
                        debug_info.append(f"artifacts: {len(result.artifacts) if result.artifacts else 0}")
                    if hasattr(result, 'message'):
                        debug_info.append(f"message: {'present' if result.message else 'none'}")
                    if debug_info:
                        server_content += f" ({', '.join(debug_info)})"
            else:
                server_content = "Invalid response structure from A2A server"
                
            logger.info(f"Received response from A2A server: {server_content[:100]}{'...' if len(server_content) > 100 else ''}")
            
            return {
                "messages": [AIMessage(content=server_content)],
                "server_response": server_content,
                "is_complete": True
            }
        
    except Exception as e:
        error_msg = f"Error communicating with A2A server: {e}"
        logger.error(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "is_complete": True
        }


def route_next_step(state: ClientAgentState) -> str:
    """Route to the next step based on current state."""
    if state.get("is_complete"):
        return END
    if not state.get("connection_ready"):
        return "initialize"
    else:
        return "send_message"


def build_client_agent_graph():
    """Build the client agent graph that communicates with A2A server."""
    
    # Create the graph
    graph = StateGraph(ClientAgentState)
    
    # Add nodes
    graph.add_node("initialize", initialize_a2a_connection)
    graph.add_node("send_message", send_to_a2a_server)
    
    # Set entry point
    graph.set_entry_point("initialize")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "initialize",
        route_next_step,
        {"send_message": "send_message", END: END}
    )
    
    graph.add_conditional_edges(
        "send_message", 
        route_next_step,
        {END: END}
    )
    
    # Compile with memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


class SimpleClientAgent:
    """Simple Client Agent that uses A2A protocol to communicate with server."""
    
    def __init__(self):
        self.graph = build_client_agent_graph()
        
    async def query(self, user_message: str) -> str:
        """Send a query to the A2A server and return the response."""
        config = {"configurable": {"thread_id": "client_session"}}
        inputs = {
            "messages": [HumanMessage(content=user_message)],
            "is_complete": False,
            "connection_ready": False,
            "server_response": ""
        }
        
        logger.info(f"Client agent processing query: {user_message}")
        
        # Stream through the graph
        final_response = ""
        async for step in self.graph.astream(inputs, config):
            logger.info(f"Graph step: {list(step.keys())}")
            for node_name, node_output in step.items():
                if "messages" in node_output:
                    latest_message = node_output["messages"][-1]
                    if isinstance(latest_message, AIMessage):
                        final_response = latest_message.content
                        
        return final_response


async def main():
    """Demo the client agent."""
    print("🤖 Simple Client Agent - A2A Demo")
    print("=" * 50)
    
    # Create client agent
    client_agent = SimpleClientAgent()
    
    # Test queries
    test_queries = [
        "What are the latest developments in artificial intelligence?",
        "Find me recent papers on transformer architectures",
        "Tell me about the benefits of renewable energy"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📤 Query {i}: {query}")
        print("-" * 30)
        
        try:
            response = await client_agent.query(query)
            print(f"📥 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 30)


if __name__ == "__main__":
    asyncio.run(main())
