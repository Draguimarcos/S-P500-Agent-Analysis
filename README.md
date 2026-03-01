# 🤖 S&P 500 Financial Analysis Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct_Agent-green?style=for-the-badge)
![Claude](https://img.shields.io/badge/Claude_3.5_Sonnet-Anthropic-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)

**An autonomous AI agent that researches and analyzes S&P 500 stocks in real time.**  
Built as a portfolio project for the [IBM RAG and Agentic AI Professional Certificate](https://www.coursera.org/professional-certificates/rag-and-agentic-ai).

</div>

---

## 📌 Overview

This project implements a **ReAct (Reasoning + Acting) agent** that performs comprehensive financial analysis on any S&P 500 company. Given a stock ticker, the agent autonomously decides which tools to use, gathers both qualitative and quantitative data, and synthesizes everything into a structured investment report.

> **No hard-coded logic.** The agent reasons dynamically — it decides *when* to search for news, *when* to pull financial metrics, and *how* to weigh them. That's the power of agentic AI.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT (Ticker)                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              REACT AGENT LOOP (LangGraph)                │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  THOUGHT │───▶│  ACTION  │───▶│   OBSERVATION    │   │
│  │          │    │          │    │                  │   │
│  │ Claude   │    │ Tool     │    │ Tool result is   │   │
│  │ reasons  │    │ is       │    │ added to context │   │
│  │ what to  │    │ called   │    │ and agent loops  │   │
│  │ do next  │    │          │    │ back to THOUGHT  │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│                                          │               │
│                          ┌───────────────┘               │
│                          │  Enough info?                 │
│                          ▼                               │
│                    ┌──────────┐                          │
│                    │  FINAL   │                          │
│                    │ ANSWER   │                          │
│                    └──────────┘                          │
└─────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌──────────────────┐      ┌──────────────────────┐
   │  TOOL 1: Tavily  │      │  TOOL 2: yfinance    │
   │  News Search     │      │  Stock Data          │
   │                  │      │                      │
   │  • Recent news   │      │  • Current price     │
   │  • Earnings      │      │  • Market cap        │
   │  • Analyst calls │      │  • P/E ratio         │
   │  • Sentiment     │      │  • 52-week range     │
   └──────────────────┘      └──────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Agent Framework** | LangGraph `create_react_agent` | Production-grade, observable state machine |
| **LLM** | Claude 3.5 Sonnet | Best-in-class reasoning for multi-step tasks |
| **News Tool** | Tavily Search | Real-time web search with financial depth |
| **Data Tool** | yfinance | Free, reliable market data — no paid API needed |
| **Temperature** | 0.1 | Low variance = consistent financial analysis |

---

## ✨ Features

- **Autonomous research loop** — agent decides how many tool calls to make
- **Real-time data** — live stock prices and breaking news
- **Structured reports** — executive summary, metrics, bull/bear analysis
- **Transparent reasoning** — prints every Thought → Action → Observation step
- **Extensible** — add more tools (SEC filings, earnings transcripts, etc.)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sp500-financial-agent.git
cd sp500-financial-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
ANTHROPIC_API_KEY=sk-ant-...    # console.anthropic.com
TAVILY_API_KEY=tvly-...         # app.tavily.com (free tier available)
```

### 5. Run the agent

```bash
# Interactive mode
python agent.py

# Pass ticker directly
python agent.py NVDA
python agent.py AAPL
python agent.py MSFT
```

---

## 📊 Example Output

```
============================================================
  🤖 Financial Agent — Analyzing: NVDA
============================================================

💭 [THOUGHT → ACTION] Calling: search_financial_news
   └─ Args: {'query': 'NVIDIA NVDA stock news earnings 2024'}

🔍 [OBSERVATION 1] Tool: search_financial_news
   └─ NVIDIA reported record quarterly revenue of $30.04B...

💭 [THOUGHT → ACTION] Calling: get_stock_data
   └─ Args: {'ticker': 'NVDA'}

🔍 [OBSERVATION 2] Tool: get_stock_data
   └─ {"symbol": "NVDA", "current_price": 875.4, "pe_ratio": 65.2...

============================================================
  ✅ Analysis Complete
============================================================

### 📊 NVIDIA Corporation (NVDA) — Financial Analysis Report

**Executive Summary**
NVIDIA continues to dominate the AI accelerator market with record revenue
driven by unprecedented demand for H100 and upcoming Blackwell GPUs...

**📈 Key Metrics**
- Current Price: $875.40 | Market Cap: $2.15T
- P/E Ratio: 65.2 | Forward P/E: 38.4
- 52-Week Range: $405.12 – $974.00
...
```

---

## 📁 Project Structure

```
sp500-financial-agent/
│
├── agent.py              # Main agent — tools, LLM, ReAct loop
├── requirements.txt      # Python dependencies
├── .env.example          # API key template
├── .env                  # Your keys (gitignored)
├── .gitignore
└── README.md
```

---

## 🔧 Extending the Agent

The tool-based architecture makes it easy to add capabilities:

```python
@tool
def get_sec_filings(ticker: str) -> str:
    """Fetches recent SEC 10-K and 10-Q filings from EDGAR."""
    # Your implementation here
    ...

# Add to the tools list
tools = [news_search_tool, get_stock_data, get_sec_filings]
```

Other ideas: earnings call transcripts, options flow, insider trading data, macro indicators.

---

## 🔑 API Keys & Cost Estimate

| Service | Free Tier | Paid |
|---|---|---|
| **Anthropic (Claude)** | $5 free credits | ~$0.003 per analysis |
| **Tavily Search** | 1,000 req/month | $0.001 per search |
| **yfinance** | ✅ Completely free | — |

A single stock analysis costs approximately **$0.01–0.02** in API calls.

---

## Learning Context

This project was built as part of the **IBM RAG and Agentic AI Professional Certificate** on Coursera. It demonstrates:

- ✅ **Agentic AI patterns** — autonomous tool use, multi-step reasoning
- ✅ **ReAct framework** — Thought → Action → Observation loops
- ✅ **Tool definition** — custom `@tool` decorators with LangChain
- ✅ **LangGraph** — stateful agent orchestration
- ✅ **Prompt engineering** — system prompts that shape agent behavior
- ✅ **API integration** — connecting multiple external services

---

# Disclaimer

This tool is for **educational purposes only**. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built using [LangGraph](https://github.com/langchain-ai/langgraph) + [Claude](https://anthropic.com) + [IBM AI Certificate](https://www.coursera.org/professional-certificates/rag-and-agentic-ai)

</div>
