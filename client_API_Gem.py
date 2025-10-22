import asyncio
import json
import re
import sys
import logging
import os
from typing import Optional, Dict, List
from contextlib import AsyncExitStack
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()  #Load the .env file

from vector import retriever


# Minimal logging setup
logging.basicConfig(level=logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)


memory_store = {}

with open("template.txt", "r", encoding="utf-8") as f:
    template = f.read()


template = template.format(
    current_date=datetime.now().strftime("%B %d, %Y (%A)")
)
# template.replace(
#                     "{current_date}", 
#                     datetime.now().strftime('%B %d, %Y (%A)')
#                 )

def get_session_history(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = ChatMessageHistory()
    return memory_store[session_id]


class MongoDBManager:
    """Handles all MongoDB operations for conversation storage"""
    
    def __init__(self, mongo_connection: str):
        # Get MongoDB connection string from command line or environment
        mongo_connection = os.environ.get('DB_connect')
        self.client = AsyncIOMotorClient(mongo_connection, server_api=ServerApi('1'))
        self.db = self.client.ease
        self.conversations = self.db.conversations
        self.bookings = self.db.bookings
        self.interactions = self.db.interactions

        
    async def test_connection(self):
        """Test MongoDB connection"""
        try:
            await self.client.admin.command('ping')
            print("✅ MongoDB connected successfully!")
            return True
        except ConnectionFailure:
            print("❌ MongoDB connection failed!")
            return False
    
    async def save_message(self, session_id: str, user_message: str, 
                          assistant_response: str, metadata: Dict = None):
        """Save a single message exchange"""
        message_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "metadata": metadata or {}
        }
        await self.interactions.insert_one(message_doc)
    
    async def save_conversation_summary(self, session_id: str, 
                                       conversation_data: Dict):
        """Save or update conversation summary"""
        await self.conversations.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "last_updated": datetime.utcnow(),
                    **conversation_data
                },
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True
        )
    
    async def save_booking(self, session_id: str, booking_data: Dict):
        """Save flight booking information"""
        booking_doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "status": booking_data.get("status", "pending"),
            "booking_details": booking_data
        }
        result = await self.bookings.insert_one(booking_doc)
        return str(result.inserted_id)
    
    async def get_conversation_history(self, session_id: str, limit: int = 50):
        """Retrieve conversation history"""
        cursor = self.interactions.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit)
        return await cursor.to_list(length=limit)
    
    async def close(self):
        """Close MongoDB connection"""
        self.client.close()


class ConversationFormatter:
    """Formats conversation data for storage and retrieval"""
    
    @staticmethod
    def extract_booking_info(conversation_text: str, tool_results: List[Dict]) -> Dict:
        """Extract structured booking information from conversation"""
        booking_info = {
            "departure_airport": None,
            "arrival_airport": None,
            "departure_date": None,
            "return_date": None,
            "trip_type": None,
            "passengers": {
                "adults": 0,
                "children": 0,
                "infants": 0
            },
            "flight_options": [],
            "selected_flight": None,
            "total_price": None,
            "currency": "XAF"
        }
        
        # Extract from tool results
        for tool_result in tool_results:
            tool_name = tool_result.get("tool_name", "")
            result_data = tool_result.get("result", {})
            
            if tool_name == "get_airport_iata_code":
                # Airport information extraction
                if isinstance(result_data, dict) and "airports" in result_data:
                    # Store airport options for reference
                    pass
            
            elif tool_name in ["one_way_flights", "round_trip_flights"]:
                # Flight search results
                booking_info["trip_type"] = "round_trip" if tool_name == "round_trip_flights" else "one_way"
                
                # Extract flight options
                if isinstance(result_data, dict):
                    booking_info["flight_options"] = result_data.get("flights", [])
                    
                    # Extract passenger info from args
                    args = tool_result.get("args", {})
                    booking_info["departure_airport"] = args.get("dep_airport_code")
                    booking_info["arrival_airport"] = args.get("arr_airport_code")
                    booking_info["departure_date"] = args.get("start_date")
                    booking_info["return_date"] = args.get("return_date")
                    booking_info["passengers"]["adults"] = args.get("adults", 0)
                    booking_info["passengers"]["children"] = args.get("children", 0)
                    booking_info["passengers"]["infants"] = args.get("infants", 0)
        
        return booking_info
    
    @staticmethod
    def format_conversation_metadata(user_input: str, response: str, 
                                    tool_results: List[Dict]) -> Dict:
        """Create metadata for conversation storage"""
        return {
            "message_length": len(user_input),
            "response_length": len(response),
            "tools_used": [t.get("tool_name") for t in tool_results],
            "has_booking_info": any(
                t.get("tool_name") in ["one_way_flights", "round_trip_flights"] 
                for t in tool_results
            )
        }


class FastMCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        mongo_connection = os.environ.get('DB_connect')

        # Get Gemini API key from parameter or environment
        model = os.environ.get('GEMINI_MODEL')
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set as environment variable")
        
        

        # Initialize ChatOpenAI with Gemini endpoint
        self.model = ChatOpenAI(
            model=model,  #the model you want for better performance
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            temperature=0.1
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | self.model
        self.conversation = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        
        # Initialize MongoDB if connection string provided
        self.db_manager = None
        if mongo_connection:
            self.db_manager = MongoDBManager(mongo_connection)
        
        self.formatter = ConversationFormatter()

    async def connect_to_server(self, server_script_path: str):
        """Connect to MCP server - minimal setup"""
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()
        print("✅ Connected to MCP Server!")
        
        # Test MongoDB connection if available
        if self.db_manager:
            await self.db_manager.test_connection()

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

        max_iterations = 10
        iteration = 0
        final_response = ""
        pending_input = user_input
        all_tool_results = []

        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Get response from LLM
                response = await self.conversation.ainvoke(
                    {"input": pending_input},
                    config={"configurable": {"session_id": session_id}}
                )

                # Extract model output
                if hasattr(response, "content"):
                    model_output = response.content
                elif isinstance(response, dict) and "content" in response:
                    model_output = response["content"]
                else:
                    model_output = str(response)

                # Look for tool calls in the response
                tool_pattern = re.compile(r"\[TOOL:\s*(\w+)\s*(\{.*?\})\]", re.DOTALL)
                matches = tool_pattern.findall(model_output)

                final_response = model_output

                if not matches:
                    # No tool calls, save to DB and return
                    if self.db_manager:
                        await self._save_conversation_to_db(
                            session_id, user_input, final_response, all_tool_results
                        )
                    return final_response

                # Execute tool calls
                tool_results = []
                for tool_name, tool_args_json in matches:
                    try:
                        tool_args = json.loads(tool_args_json)
                    except json.JSONDecodeError:
                        tool_args = {}

                    tool_result = await self.execute_tool_call(tool_name, tool_args)
                    tool_results.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "result": tool_result
                    })
                    all_tool_results.append({
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
                    if tool_info['result'].get('success', True):
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
                    pending_input = (
                        f"Multiple tools were executed. Results:\n"
                        f"{json.dumps(tool_results, indent=2)}\n\n"
                        f"Extract information from this data(link if there is any) and provide it to the user in a more readable format without any deletion of information."
                    )

            except Exception as e:
                error_response = f"I encountered an error while processing your request: {str(e)}"
                if self.db_manager:
                    await self._save_conversation_to_db(
                        session_id, user_input, error_response, all_tool_results
                    )
                return error_response

        if iteration >= max_iterations:
            final_response = final_response + "\n\n(Note: Response may be incomplete due to processing limits)"

        # Save final conversation to DB
        if self.db_manager:
            await self._save_conversation_to_db(
                session_id, user_input, final_response, all_tool_results
            )

        # final_response = re.sub(r"\[TOOL:[^\]]*\]", "", final_response).strip()
        return final_response

    async def _save_conversation_to_db(self, session_id: str, user_input: str, 
                                      response: str, tool_results: List[Dict]):
        """Save conversation and extracted booking info to MongoDB"""
        if not self.db_manager:
            return
        
        try:
            # Format metadata
            metadata = self.formatter.format_conversation_metadata(
                user_input, response, tool_results
            )
            
            # Save individual message
            await self.db_manager.save_message(
                session_id, user_input, response, metadata
            )
            
            # Extract and save booking information if available
            booking_info = self.formatter.extract_booking_info(response, tool_results)
            
            if booking_info.get("flight_options") or booking_info.get("departure_airport"):
                booking_info["status"] = "searching" if booking_info.get("flight_options") else "pending"
                await self.db_manager.save_booking(session_id, booking_info)
            
            # Update conversation summary
            conversation_summary = {
                "session_id": session_id,
                "message_count": len(memory_store.get(session_id, ChatMessageHistory()).messages),
                "last_message": user_input[:100],
                "booking_info": booking_info
            }
            await self.db_manager.save_conversation_summary(session_id, conversation_summary)
            
        except Exception as e:
            print(f"Warning: Failed to save to database: {str(e)}")

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
            if self.db_manager:
                await self.db_manager.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 client_API_Gem.py server.py")
        sys.exit(1)

    
    # # Get MongoDB connection string from command line or environment
    # mongo_connection = os.environ.get('DB_connect')
    
    # # Get Gemini API key from command line or environment
    # gemini_key = os.environ.get('GEMINI_API_KEY')

    client = FastMCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat("user1")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())