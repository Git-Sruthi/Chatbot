import streamlit as st
import json
import os
import requests
from datetime import datetime
import re
import faiss
import pickle
import pdfplumber
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
ALPHA_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Load user data
with open("data.json", "r") as file:
    data = json.load(file)
user = data["user"]

# Extract potential stock symbols

def extract_possible_symbol(user_input):
    return re.findall(r'\b[A-Z]{1,5}\b', user_input)

# Call Together AI
def call_together_ai(prompt):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    return result["choices"][0]["message"]["content"]

# Get stock price using Alpha Vantage
def get_stock_price(symbol):
    symbol = symbol.upper()
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if "Time Series (Daily)" not in data:
            return f"Sorry, I couldn't find stock data for '{symbol}'."

        time_series = data["Time Series (Daily)"]
        dates = sorted(time_series.keys(), reverse=True)
        today_data = time_series[dates[0]]
        prev_data = time_series[dates[1]]

        open_price = float(today_data["1. open"])
        high_price = float(today_data["2. high"])
        low_price = float(today_data["3. low"])
        close_price = float(today_data["4. close"])
        volume = int(float(today_data["5. volume"]))
        prev_close = float(prev_data["4. close"])
        change = close_price - prev_close
        percent = (change / prev_close) * 100

        invested = symbol in user.get("portfolio", [])
        date_fmt = datetime.strptime(dates[0], "%Y-%m-%d").strftime("%B %d, %Y")

        reply = f"""**📈 {symbol} Stock Performance on {date_fmt}:**

- **Opening Price**: ${open_price:.2f}  
- **Daily High / Low**: ${high_price:.2f} / ${low_price:.2f}  
- **Closing Price**: ${close_price:.2f}  
- **Previous Close**: ${prev_close:.2f}  
- **Change**: ${change:+.2f} ({percent:+.2f}%)  
- **Volume**: {volume:,} shares traded  

{ "✅ You have invested in this stock." if invested else "⚠️ You do not currently hold shares of this stock." }

_Always do your research or consult a financial advisor before making decisions._
"""
        return reply

    except Exception:
        return "Sorry, I couldn't fetch the stock data right now."

# PDF Upload & Embedding
def process_pdf(uploaded_file):
    if uploaded_file:
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return full_text
    return ""

# Initialize Streamlit app
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("Chatbot At Your Service")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        ("Bot", f"Hi {user['name']}, I’m your assistant. How can I help you today?")
    ]

uploaded_file = st.file_uploader("Upload a financial PDF", type="pdf")
pdf_text = process_pdf(uploaded_file) if uploaded_file else ""

for sender, message in st.session_state.chat_history:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("You", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    if pdf_text and any(q in user_input.lower() for q in ["explain", "summarize", "what does", "meaning"]):
        prompt = f"""A user has uploaded a financial document. The user question is:
"{user_input}"

Refer to the content below and explain in a simple way:

{pdf_text[:2000]}

Answer:
"""
        bot_reply = call_together_ai(prompt)
    else:
        input_lower = user_input.lower()
        if "bank balance" in input_lower:
            bot_reply = f"Your current bank balance is ₹{user['bank_balance']:,}."
        elif "email" in input_lower:
            bot_reply = f"Your registered email is {user['email']}."
        elif "my name" in input_lower:
            bot_reply = f"You're {user['name']}."
        elif "what stocks do i own" in input_lower:
            portfolio = user.get("portfolio", [])
            bot_reply = "You're currently invested in:\n" + "\n".join([f"✅ {sym}" for sym in portfolio]) if portfolio else "You currently don't own any stocks."
        elif any(word in input_lower for word in ["stock", "price", "performance"]):
            symbols = extract_possible_symbol(user_input)
            bot_reply = get_stock_price(symbols[0]) if symbols else "Couldn't find a valid stock symbol in your query."
        else:
            bot_reply = call_together_ai(user_input)

    st.session_state.chat_history.append(("Bot", bot_reply))
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
