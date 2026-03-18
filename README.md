# AI Financial Analyst Agent 🤖📈

An AI-powered financial analysis tool that answers natural language queries on real-time stock data using LLaMA 3.3 70B via Groq API.

🔗 **Live Demo:** [AI Financial Analyst Agent](https://ai-financial-analyst-agent-mv5xxqjaroaxfjtzparfg2.streamlit.app/)

---

## What It Does

Enter a stock ticker and ask any financial question — the AI analyzes real OHLCV data and responds like a professional analyst.

**Example queries:**
- "Is TSLA trading above its 50-day MA?"
- "Is there a crossover between MA_10 and MA_50 for NVDA?"
- "Give me a summary of AAPL's recent performance"
- "Is AMZN showing signs of a trend reversal?"

---

## Tech Stack

| Layer | Tools |
|---|---|
| Frontend & Deployment | Python, Streamlit |
| AI Inference | Groq API (LLaMA 3.3 70B) |
| Data Pipeline | yFinance, Pandas |
| Hosting | Streamlit Community Cloud |

---

## How It Works

1. User inputs a stock ticker and a natural language question
2. App fetches real-time OHLCV data via yFinance
3. Computes MA_10 and MA_50 technical indicators
4. Injects a 20-row rolling window with statistical summary into the LLM prompt
5. LLaMA 3.3 70B via Groq returns a professional financial insight

---

## Run Locally
```bash
git clone https://github.com/suyashwagh1/AI-Financial-Analyst-Agent.git
cd AI-Financial-Analyst-Agent
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

---

## Project Structure
```
├── app.py              # Streamlit UI
├── agent.py            # AI agent logic
├── finance_utils.py    # Data pipeline & indicators
├── requirements.txt
└── .gitignore
```

---

## Author

**Suyash Wagh**
[LinkedIn](https://www.linkedin.com/in/suyashwagh11) | [GitHub](https://github.com/suyashwagh1)
