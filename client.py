import asyncio
import json
import re
import sys
import logging
from typing import Optional
from contextlib import AsyncExitStack
from datetime import datetime, timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from vector import retriever
from typing import List, Optional, Dict

# Minimal logging setup
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("ollama").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

template = """
You are a friendly multilingual virtual travel assistant for EASE TRAVEL AGENCY. Your primary role is to help users book plane tickets to their destinations..

Response language: Always match the user's input language
Tone: Conversational and natural (avoid showing internal reasoning to users)

Current date: """ + datetime.now().strftime('%B %d, %Y (%A)') + """

Core Objectives:

- Help users book flights efficiently
- Ask clarifying questions before confirming bookings
- Provide clear, well-structured responses
- Convert relative dates to specific dates (DD-MM-YYYY format)

Here are the informations you need to have from users:
- “Valid Departure city”
- "Valid Destination"
- "Departure date" (convert relative dates to actual dates: DD-MM-YYYY format)
- “Is this a round-trip flight?”
- "If round-trip: what is the return date?" (convert relative dates to actual dates)
- “Number of people” : “Are they adults or children or infants”

CRITICAL WORKFLOW - For tool calling:

Use format: [TOOL: tool_name JSON_ARGS] like [TOOL: get_airport_iata_code {{"city": "Paris"}}])

1. **STEP 1 - GET AIRPORTS**:

- When user mentions cities, immediately call get_airport_iata_code tool (e.g User says "Paris" → [TOOL: get_airport_iata_code {{"city": "Paris"}}])
- Extract important informations from the results and present options naturally to the user (don't mention using tools)
- Get user's airport selection before proceeding

2. **STEP 2 - COLLECT ALL INFO**: Gather these details: 

- Confirmed departure airport code (from tool results only) 
- Confirmed arrival airport code (from tool results only) 
- Departure date (DD-MM-YYYY format) 
- Number of travelers (number of adults, number of children, number of infant) 
- Return date if round trip

3. **STEP 3 - MANDATORY CONFIRMATION**: Before calling search_flights, ALWAYS show user: 
"I will search for flights with this information: 
✈️ Departure: [AIRPORT_NAME] ([IATA Code]) 
✈️ Arrival: [AIRPORT_NAME] ([IATA Code]) 
📅 Depart_date: [START DATE] 
👥 Passengers: [x] adults, [x] children, [x] infants 
📅 return_date: [RETURN DATE] if round trip


ask user to confirm information

4. **STEP 4 - FLIGHT SEARCH**: ONLY after user's confirmation, call: 
- [TOOL: one_way_flights {{"dep_airport_code": "AIRPORT_NAME-CODE", "arr_airport_code": "AIRPORT_NAME-CODE", "start_date": "START DATE", 
"adults": x, "children": x, "infants": x}}] if it is a one way trip 
- [TOOL: round_trip_flights {{"dep_airport_code": "AIRPORT_NAME-CODE", "arr_airport_code": "AIRPORT_NAME-CODE", 
"start_date": "START DATE", "return_date": "RETURN DATE", "adults": x, "children": x, "infants": x}}] if it is a round trip

5 ** STEP 5 : Results Presentation
- If the currency is not XAF convert it in XAF.
- Make sure to present all flight available and informations from API response with more possible details even link in a more user-friendly way(never show wich tools you have used)
- Format clearly with airlines, times, prices in Franc cfa, Link  
- Never modify or omit API data
- If no results, inform user clearly

Conversation Management Rules

Information Updates:

- ADD: New information to existing booking information
- UPDATE: Changed information without losing other details
- MAINTAIN: Complete booking history throughout conversation
- SUMMARIZE: Current booking status after each update

Missing Information

- Ask targeted questions for missing details only
- Don't request information you already have

Date Conversion Reference
Based on current date, convert:

- "Friday" → Next Friday (YYYY-MM-DD)
- "next Monday" → Following Monday (YYYY-MM-DD)
- "in 5 days" → Current date + 5 days (YYYY-MM-DD)
- "first Sunday in September" → September's first Sunday (YYYY-MM-DD)

Always confirm converted dates with user

Critical Constraints

NEVER DO:

- ❌ Invent flight details, prices, or airline information
- ❌ Call search tools without explicit user confirmation
- ❌ Assume airport codes - always use tool results
- ❌ Mention tool usage to users (present info naturally , Avoid message like "The tool 'get_airport_iata_code'" or  "The tool 'round_trip_flights'"  even "The tool 'one_way_flights'")
- ❌ Delete or restart booking context unnecessarily

ALWAYS DO:

- ✅ Use exact airport codes from tool responses
- ✅ Show complete booking summary after changes
- ✅ Ask for clarification when requests are ambiguous
- ✅ Confirm converted dates with users
- ✅ Display all real flight data from API responses


Error Handling

- No airport code found: Inform user and suggest alternative cities
- No flights found: Inform user and suggest alternative dates/airports
- Tool errors: Ask user to retry or provide alternative
- Invalid dates: Request clarification with valid format

Response Quality Standards

- Use bullet points or tables for flight options
- Include all relevant details (times, prices, airlines)
- Be concise but complete
- Maintain professional yet friendly tone

"""

memory_store = {}



def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]

class FastMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.model = ChatOllama(model="llama3.1:8b", temperature=0.0) # ollama run  llama3.1:latest

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("system", "Context:\n{context}"),
            ("human", "{input}")
        ])

        chain = prompt | self.model
        self.conversation = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server - minimal setup"""
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print("✅ Connected!")

    async def execute_tool_call(self, tool_name: str, tool_args: dict) -> dict:
        """Execute a single tool call with proper error handling"""
        try:
        
            tool_result = await self.session.call_tool(tool_name, tool_args)
            
            # Handle different types of tool result content
            if hasattr(tool_result, 'content'):
                if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                    # MCP often returns content as a list of TextContent objects
                    content = tool_result.content
                    if hasattr(content, 'text'):
                        result_text = content.text
                    else:
                        result_text = str(content)
                else:
                    result_text = str(tool_result.content)
            else:
                result_text = str(tool_result)
            
            # Try to parse as JSON if possible
            try:
                return json.loads(result_text)
            except (json.JSONDecodeError, TypeError):
                return {"raw_content": result_text}
                
        except Exception as e:
            return f"Tool error: {str(e)}"
  

    async def process_query(self, user_input: str, session_id: str = "default") -> str:
        """
        Process user query with improved error handling and response formatting.
        """
        try:
            context = retriever.invoke(user_input)
        except Exception as e:
            context = ""

        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        final_response = ""
        pending_input = user_input

        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Get response from LLM
                response = await self.conversation.ainvoke(
                    {"input": pending_input, "context": context},
                    config={"configurable": {"session_id": session_id}}
                )

                # Extract model output
                if hasattr(response, "content"):
                    model_output = response.content
                elif isinstance(response, dict) and "content" in response:
                    model_output = response["content"]
                    print(model_output)
                else:
                    model_output = str(response)
                    print(model_output)

                # Look for tool calls in the response
                tool_pattern = re.compile(r"\[TOOL:\s*(\w+)\s*(\{.*?\})\]", re.DOTALL)
                matches = tool_pattern.findall(model_output)

                final_response = model_output

                
                if not matches:
                    # No tool calls, this is the final response
                    return final_response
                

                # Execute tool calls
                tool_results = []
                for tool_name, tool_args_json in matches:
                    try:
                        tool_args = json.loads(tool_args_json)
                    except json.JSONDecodeError as e:
                        tool_args = {}

                    # Execute the tool
                    tool_result = await self.execute_tool_call(tool_name, tool_args)
                    tool_results.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": tool_result
                    })

                # Add tool results to conversation history
                session_history = get_session_history(session_id)
                for tool_info in tool_results:
                    result_message = (
                        f"Tool '{tool_info['tool_name']}' executed with args {tool_info['args']}. "
                        f"Result: {json.dumps(tool_info['result'], indent=2)}"
                    )
                    session_history.add_user_message(result_message)

                # Prepare next iteration input
                if len(tool_results) == 1:
                    tool_info = tool_results[0]
                    if tool_info['result'].get('success', True):  # Default to True for backward compatibility
                        pending_input = (
                            f"The tool '{tool_info['tool_name']}' returned: "
                            f"{json.dumps(tool_info['result'], indent=2)}\n\n"
                            f"Extract information from this data(link if there is any) and provide it to the user in a more readable format without any deletion of information. "
                            f"If there were any errors, explain them clearly and suggest alternatives."
                        )
                    else:
                        pending_input = (
                            f"The tool '{tool_info['tool_name']}' failed with error: "
                            f"{tool_info['result'].get('error', 'Unknown error')}\n\n"
                            f"Please explain this error to the user and suggest what they should do next."
                        )
                else:
                    # Multiple tool results
                    pending_input = (
                        f"Multiple tools were executed. Results:\n"
                        f"{json.dumps(tool_results, indent=2)}\n\n"
                        f"Extract information from this data(link if there is any) and provide it to the user in a more readable format without any deletion of information."
                    )

            except Exception as e:
                return f"I encountered an error while processing your request: {str(e)}"

        if iteration >= max_iterations:
            return final_response + "\n\n(Note: Response may be incomplete due to processing limits)"

        final_response = re.sub(r"\[TOOL:[^\]]*\]", "", final_response).strip()

        
        print(tool_results)
        return final_response

    

    async def chat(self, session_id: str = "default"):
        """Interactive chat loop with improved error handling."""
        print("🚀 MCP Travel Assistant started!")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    print("\n Thank you for using EASE TRAVEL AGENCY! Have a great day!")
                    break
                
                if not user_input:
                    continue

                response = await self.process_query(user_input, session_id=session_id)

                print(f"\n Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nSorry, I encountered an error: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        try:
            await self.exit_stack.aclose()
        except Exception as e:
            print(f"Error: {e}")



async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py server.py")
        sys.exit(1)

    client = FastMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat("user1")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())