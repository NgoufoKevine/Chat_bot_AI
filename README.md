# ✈️ Airline Ticket Chatbot  

## 📌 Project Overview  
This project implements a **conversational chatbot** capable of offering **airline tickets for a travel agency**.  
(July 2025).  

The chatbot leverages **Large Language Models (LLMs)** , MCP Server and integrates with **Meta’s WhatsApp Business API** to enable natural and interactive communication with users.  

---

## 🚀 Features  
- 🗣️ **Conversational AI**: Handles natural language queries from users.  
- 👥 **Dynamic Context Updates**: Remembers user-provided details (e.g., passengers, dates, destinations) and updates them without restarting the whole search.  
- 📅 **Flight Search**: Retrieves and proposes flight options based on user preferences.  
- 🔄 **Webhook Integration**: Connected to Meta Developers for real-time message exchange.  
- 🧪 **User Testing Framework**: Designed with a focus on human–AI interaction for better usability.  

---

## 🛠️ Tech Stack  
- **Language Models**: LLMs for natural language understanding.  
- **Backend**: Python (Flask/FastAPI).  
- **Database**: SQLite / PostgreSQL for user session management.  
- **API Integration**: Meta WhatsApp Business API for communication.  
- **Deployment**: Docker & Ngrok for testing, cloud deployment planned.  

---

## ⚙️ Installation  

1. **Clone the repository**  
```bash
git clone https://github.com/NgoufoKevine/Chat_bot_AI.git
cd Chat_bot_AI 
```

2. **Set up a virtual environment**  
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**  
Create a `.env` file and set your credentials:  
```
META_ACCESS_TOKEN=your_token_here
VERIFY_TOKEN=your_webhook_verify_token
APP_SECRET=your_app_secret
```

5. **Run the server**  
```bash
python server.py
```

---

## 📡 Webhook Setup  
- Create a **Meta Developer App** and configure **Webhook URL + Verify Token**.  
- Subscribe to the **messages** endpoint.  
- Replace temporary tokens with a **permanent access token** generated via **system user**.  

---

## 📅 Context  
This project is part of my professional internship (July–December 2025).  
Supervisor requested: **(open for collaboration/mentorship, especially on LLMs & AI agents)**.  

---

## 📌 Future Work  
- 🔍 Integration with live airline APIs (Amadeus, Skyscanner).  
- 🧠 Fine-tuning of LLM for travel-specific intents.  
- 🌍 Multi-language support (English, French).  
- 📊 Evaluation with real user interactions.  

---

## 👨‍💻 Author  
**Kevine Grace**  
- MSc Data Science (AIMS, Cameroon)  
- MSc Physics (Electronics, Electrical Engineering & Automation, University of Dschang)  
- Passion: AI Agents 🤖, Human–AI Interaction, Data-Driven Systems  
