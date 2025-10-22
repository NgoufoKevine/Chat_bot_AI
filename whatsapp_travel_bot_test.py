import asyncio
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Charge le fichier .env

# Import your existing client
from client_API_Gem import FastMCPClient


app = FastAPI()

WHATSAPP_VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN')
WHATSAPP_ACCESS_TOKEN = os.environ.get('WHATSAPP_ACCESS_TOKEN')
PHONE_NUMBER_ID = os.environ.get('PHONE_NUMBER_ID')

WHATSAPP_API_URL = f"https://graph.facebook.com/v17.0/{PHONE_NUMBER_ID}/messages"

# Initialize the MCP client globally
mcp_client = None

class WhatsAppMessage(BaseModel):
    object: str
    entry: list

async def initialize_client():
    """Initialize the MCP client on startup"""
    global mcp_client
    if mcp_client is None:
        mcp_client = FastMCPClient()
        # Connect to your server - adjust path as needed
        await mcp_client.connect_to_server("server.py")
        print("✅ MCP Client initialized and connected!")

async def send_whatsapp_message(to: str, message: str):
    """Send message back to WhatsApp"""
    headers = {
        'Authorization': f'Bearer {WHATSAPP_ACCESS_TOKEN}',
        'Content-Type': 'application/json',
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    
    try:
        response = requests.post(WHATSAPP_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending WhatsApp message: {e}")
        return None

def schedule_task(coro):
    """Run background task and log errors"""
    task = asyncio.create_task(coro)

    def _on_done(t):
        try:
            t.result()
        except Exception as e:
            print(f"❌ Exception in background task: {e}", flush=True)

    task.add_done_callback(_on_done)
    return task

@app.on_event("startup")
async def startup_event():
    """Initialize the client when the FastAPI app starts"""
    await initialize_client()

@app.get("/")
def read_root():
    return {"message": "Travel Assistant WhatsApp Bot is running!"}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """WhatsApp webhook verification (keep your existing logic)"""
    try:
        query_params = request.query_params
        print('-----------------------------------------', query_params)
        
        mode = query_params.get("hub.mode")
        verify_token = query_params.get("hub.verify_token")
        challenge = query_params.get("hub.challenge")
        
        if mode == "subscribe" and verify_token == WHATSAPP_VERIFY_TOKEN:
            print("✅ Vérification réussie - Token valide")
            return Response(content=challenge, status_code=200)
        else:
            print("❌ Vérification échouée - Token invalide ou mode incorrect")
            return Response(content="Forbidden", status_code=403)
        
    except Exception as e:
        print(f"❌ Erreur verify_token: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/webhook")
async def handle_whatsapp_message(message_data: dict):
    """Handle incoming WhatsApp messages"""
    try:
        schedule_task(process_whatsapp_event(message_data))
        return {"status": "received"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def process_whatsapp_event(message_data: dict):
    """Process WhatsApp event asynchronously"""
    global mcp_client
    try:
        print("📥 Incoming WhatsApp payload:", message_data, flush=True)

        if mcp_client is None:
            print("⚡ MCP client not initialized, reconnecting...", flush=True)
            await initialize_client()

        if message_data.get("object") == "whatsapp_business_account":
            for entry in message_data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") == "messages":
                        value = change.get("value", {})
                        messages = value.get("messages", [])

                        for message in messages:
                            sender = message.get("from")
                            message_text = message.get("text", {}).get("body", "")
                            message_type = message.get("type")

                            if message_type == "text" and message_text:
                                print(f"📱 Received message from {sender}: {message_text}", flush=True)

                                response = await mcp_client.process_query(
                                    user_input=message_text,
                                    session_id=sender
                                )

                                print(f"🤖 Assistant response: {response}", flush=True)
                                await send_whatsapp_message(sender, response)
    except Exception as e:
        print(f"❌ Error in process_whatsapp_event: {e}", flush=True)

@app.post("/test-message")
async def test_message(test_data: dict):
    """Test endpoint to simulate a message without WhatsApp"""
    try:
        global mcp_client
        
        if mcp_client is None:
            await initialize_client()
        
        user_input = test_data.get("message", "Hello")
        session_id = test_data.get("session_id", "test_user")
        
        response = await mcp_client.process_query(
            user_input=user_input,
            session_id=session_id
        )
        
        return {
            "user_input": user_input,
            "assistant_response": response,
            "session_id": session_id
        }
    
    except Exception as e:
        return {"error": str(e)}

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up when the app shuts down"""
    global mcp_client
    if mcp_client:
        await mcp_client.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
