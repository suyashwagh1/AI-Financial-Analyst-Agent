import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from finance_utils import create_data_summary

load_dotenv()

api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found.")

client = Groq(api_key=api_key)


def ai_finance_agent(user_query, df, ticker):
    summary = create_data_summary(df)
    recent_data = df.tail(20).to_string(index=False)

    prompt = f"""
You are an expert financial analyst.

Ticker: {ticker}

Dataset summary:
{summary}

Most recent rows of data:
{recent_data}

User question:
{user_query}

Instructions:
- Focus on the latest available dates in the dataset
- Mention the actual latest date you see
- Do not talk about dates that are not present in the dataframe
- Keep the answer concise and professional
"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    return chat_completion.choices[0].message.content