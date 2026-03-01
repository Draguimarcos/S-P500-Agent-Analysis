"""
╔══════════════════════════════════════════════════════════════╗
║        S&P 500 Financial Analysis Agent                      ║
║        IBM RAG & Agentic AI Certification — Portfolio Project║
╚══════════════════════════════════════════════════════════════╝

Architecture: ReAct (Reasoning + Acting)
Flow: Thought → Action → Observation → ... → Final Answer
LLM: Claude 3.5 Sonnet via Anthropic API
Framework: LangGraph (StateGraph)
"""

import os
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

import yfinance as yf
import json

# ─────────────────────────────────────────────
# 1. LOAD ENVIRONMENT VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")

if not ANTHROPIC_API_KEY or not TAVILY_API_KEY:
    raise EnvironmentError(
        "Missing API keys. Please check your .env file.\n"
        "Required: ANTHROPIC_API_KEY and TAVILY_API_KEY"
    )


# ─────────────────────────────────────────────
# 2. TOOL DEFINITIONS
#    Each tool represents one "Action" the
#    agent can take during its reasoning loop.
# ─────────────────────────────────────────────

@tool
def get_stock_data(ticker: str) -> str:
    """
    Fetches real-time and historical financial data for a given stock ticker.
    Returns: current price, market cap, P/E ratio, 52-week range,
             volume, EPS, dividends and 5-day price history.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info  = stock.info

        # Extract the most relevant financial metrics
        data = {
            "symbol":            info.get("symbol", ticker.upper()),
            "company_name":      info.get("longName", "N/A"),
            "current_price":     info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
            "previous_close":    info.get("previousClose", "N/A"),
            "market_cap":        info.get("marketCap", "N/A"),
            "pe_ratio":          info.get("trailingPE", "N/A"),
            "forward_pe":        info.get("forwardPE", "N/A"),
            "eps":               info.get("trailingEps", "N/A"),
            "52_week_high":      info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low":       info.get("fiftyTwoWeekLow", "N/A"),
            "volume":            info.get("volume", "N/A"),
            "avg_volume":        info.get("averageVolume", "N/A"),
            "dividend_yield":    info.get("dividendYield", "N/A"),
            "beta":              info.get("beta", "N/A"),
            "sector":            info.get("sector", "N/A"),
            "industry":          info.get("industry", "N/A"),
            "analyst_target":    info.get("targetMeanPrice", "N/A"),
            "recommendation":    info.get("recommendationKey", "N/A"),
        }

        # Fetch last 5 trading days for price trend context
        history = stock.history(period="5d")
        if not history.empty:
            data["5_day_prices"] = {
                str(date.date()): round(close, 2)
                for date, close in history["Close"].items()
            }

        return json.dumps(data, indent=2, default=str)

    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"


# Tavily provides real-time web search — ideal for recent news
# This is the agent's "knowledge retrieval" action
news_search_tool = TavilySearchResults(
    max_results=4,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    name="search_financial_news",
    description=(
        "Searches the web for recent financial news, earnings reports, analyst ratings, "
        "and market sentiment about a specific company or stock ticker. "
        "Use this BEFORE get_stock_data to understand qualitative context."
    ),
)


# ─────────────────────────────────────────────
# 3. LLM CONFIGURATION
#    Claude 3.5 Sonnet is the "brain" of the
#    agent — it decides WHEN and HOW to use tools.
# ─────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    anthropic_api_key=ANTHROPIC_API_KEY,
    temperature=0.1,   # Low temperature for consistent financial analysis
    max_tokens=4096,
)

# ─────────────────────────────────────────────
# 4. SYSTEM PROMPT — AGENT PERSONA & RULES
#    This shapes the agent's reasoning strategy
#    and the structure of its final output.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Financial Analysis Agent specializing in S&P 500 equities.

Your reasoning process MUST follow the ReAct pattern:
  1. THOUGHT  — Analyze what information you need and why
  2. ACTION   — Call the appropriate tool(s) to gather data
  3. OBSERVATION — Interpret the results before proceeding
  4. Repeat until you have enough information for a comprehensive answer

## RESEARCH PROTOCOL
For every stock analysis request:
  Step 1: Use `search_financial_news` to find recent news, sentiment, and analyst opinions
  Step 2: Use `get_stock_data` to retrieve quantitative financial metrics
  Step 3: Synthesize both qualitative and quantitative data into a structured report

## OUTPUT FORMAT
Structure your final response as follows:

### 📊 [COMPANY NAME] ([TICKER]) — Financial Analysis Report

**Executive Summary** — 2-3 sentence overview of current position

**📈 Key Metrics**
- Price & Valuation (current price, P/E, market cap)
- Performance (52-week range, beta, volume)
- Analyst Consensus (target price, recommendation)

**📰 Recent News & Catalysts** — Key developments affecting the stock

**⚖️ Bull vs Bear Case** — Balanced perspective on upside and downside risks

**🎯 Investment Considerations** — Key factors for investors to monitor

**⚠️ Disclaimer:** This analysis is for educational purposes only and does not constitute financial advice.

Be precise, data-driven, and objective. Always cite the source of your data (live market data vs news search).
"""


# ─────────────────────────────────────────────
# 5. AGENT CONSTRUCTION
#    create_react_agent from LangGraph builds
#    the Thought → Action → Observation loop
#    automatically using the tools we defined.
# ─────────────────────────────────────────────
tools = [news_search_tool, get_stock_data]

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=SYSTEM_PROMPT,  # Injects system prompt into agent state
)


# ─────────────────────────────────────────────
# 6. AGENT RUNNER
#    Streams the agent's output so we can see
#    the Thought → Action → Observation steps.
# ─────────────────────────────────────────────
def analyze_stock(ticker: str, verbose: bool = True) -> str:
    """
    Runs the financial analysis agent for a given stock ticker.

    Args:
        ticker:  Stock ticker symbol (e.g. 'AAPL', 'GOOGL', 'TSLA')
        verbose: If True, prints intermediate reasoning steps

    Returns:
        Final analysis report as a string
    """
    print(f"\n{'='*60}")
    print(f"  🤖 Financial Agent — Analyzing: {ticker.upper()}")
    print(f"{'='*60}\n")

    user_message = (
        f"Please perform a comprehensive financial analysis of {ticker.upper()}. "
        f"Search for recent news first, then retrieve the stock data, "
        f"and provide a full investment analysis report."
    )

    # ── AGENT EXECUTION LOOP ──────────────────────────────────
    # LangGraph streams each step: tool calls and LLM responses
    final_response = ""
    step_count = 0

    for chunk in agent.stream(
        {"messages": [HumanMessage(content=user_message)]},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])
        if not messages:
            continue

        last_msg = messages[-1]
        msg_type = type(last_msg).__name__

        # ── OBSERVATION: Show what tools returned ──────────────
        if verbose and hasattr(last_msg, "content"):
            if msg_type == "ToolMessage":
                step_count += 1
                print(f"🔍 [OBSERVATION {step_count}] Tool: {last_msg.name}")
                content_preview = str(last_msg.content)[:200]
                print(f"   └─ {content_preview}...\n")

            # ── THOUGHT + ACTION: Show agent reasoning ─────────
            elif msg_type == "AIMessage":
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        print(f"💭 [THOUGHT → ACTION] Calling: {tc['name']}")
                        args_preview = str(tc.get("args", {}))[:100]
                        print(f"   └─ Args: {args_preview}\n")
                elif last_msg.content:
                    # This is the final response
                    final_response = last_msg.content

    print("\n" + "="*60)
    print("  ✅ Analysis Complete")
    print("="*60 + "\n")

    return final_response


# ─────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # Accept ticker as command-line argument or prompt user
    if len(sys.argv) > 1:
        ticker_input = sys.argv[1]
    else:
        ticker_input = input("Enter stock ticker to analyze (e.g. AAPL, MSFT, NVDA): ").strip()

    if not ticker_input:
        print("Error: Please provide a valid ticker symbol.")
        sys.exit(1)

    result = analyze_stock(ticker_input, verbose=True)
    print(result)
