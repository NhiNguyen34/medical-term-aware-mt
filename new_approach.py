import json
import csv
import os
import logging
from turtle import pd
from typing import Dict, List, Optional, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import openai
from openai import RateLimitError
from threading import Lock
from googleapiclient.discovery import build
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import random
# Setup logging
logging.basicConfig(
    filename="agent_run.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

USE_SERPAPI = True  # bật True nếu bạn muốn dùng serpapi

if USE_SERPAPI:
    from serpapi import GoogleSearch
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    if SERPAPI_API_KEY is None:
        raise RuntimeError("❌ Bạn cần đặt SERPAPI_API_KEY")

# OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise RuntimeError("❌ Bạn cần đặt OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
from tavily import TavilyClient
import requests
from ddgs import DDGS
ddgs_lock = Lock()
ddgs_client = DDGS()
_ddgs = None

def get_ddgs():
    global _ddgs
    if _ddgs is None:
        _ddgs = DDGS()  # mở 1 lần
    return _ddgs

google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
google_cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")  # your custom search engine ID

class AzureLLMClient:
    """Azure OpenAI client with structured output support using modern tools parameter"""

    def __init__(self,
                 deployment_key=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                 api_version_key=os.getenv("AZURE_OPENAI_API_VERSION"),
                 endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                 api_key_key=os.getenv("AZURE_OPENAI_API_KEY"),
                 temperature: float = 0.8):
        self.client = AzureChatOpenAI(
            azure_deployment=deployment_key,
            api_version=api_version_key,
            temperature=temperature,
            max_retries=20,
            max_completion_tokens=16384,
            top_p=0.95,
            azure_endpoint=endpoint,
            api_key=api_key_key,
            stream_usage=True
        )

        self.fallback_client = ChatOpenAI(
            model_name="gpt-4.1",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 0.1,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        max_retries: int = 1,
        base_delay: float = 1.0
    ) -> str:
        """
        Standard chat completion with modern tools parameter and retry logic
        """
        attempt = 0

        # for attempt in range(max_retries):
        try:
            # For function calling, use bind_tools and proper tool_choice
            if tools:
                # Bind tools to the client for function calling
                llm_with_tools = self.client.bind_tools(tools)

                # If tool_choice is specified, force that specific function
                if tool_choice:
                    # Format tool_choice for forcing specific function
                    if isinstance(tool_choice, str):
                        # If just function name is provided
                        formatted_tool_choice = {
                            "type": "function",
                            "function": {
                                "name": tool_choice
                            }
                        }
                    elif isinstance(tool_choice, dict) and "function" in tool_choice:
                        # If already properly formatted
                        formatted_tool_choice = tool_choice
                    else:
                        formatted_tool_choice = tool_choice

                    # Use the proper kwargs for forcing function call
                    response = llm_with_tools.invoke(
                        messages,
                        # reasoning_effort="high",
                        tool_choice=formatted_tool_choice
                    )
                else:
                    response = llm_with_tools.invoke(messages)
            else:
                # Regular chat completion without tools
                response = self.client.invoke(messages)

            # Handle response - check for tool calls first
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Return the function arguments from the first tool call
                tool_call = response.tool_calls[0]
                if hasattr(tool_call, 'args'):
                    return json.dumps(tool_call.args)
                else:
                    # If no args attribute, try to extract from tool call dict representation
                    tool_call_dict = tool_call if isinstance(tool_call, dict) else tool_call.__dict__
                    return json.dumps(tool_call_dict)
            elif hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:

        
            try:
                openai_response = self.fallback_client.invoke(messages)
                logging.info("🔄 Fallback to OpenAI succeeded")
                return getattr(openai_response, "content", str(openai_response))
            except Exception as e2:
                logging.error(f"❌ OpenAI fallback also failed: {e2}")

            attempt += 1
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise RuntimeError("Both Azure and OpenAI completions failed after retries")


            # if attempt < max_retries - 1:
                # Exponential backoff with jitter
            # delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            # logger.info(f"Retrying in {delay:.1f}s...")
            # time.sleep(delay)
            # else:
            #     logger.error(f"All {max_retries} attempts failed")
            #     raise

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = 1,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Completion with structured output using modern tools parameter (2025)
        """

        # Force the specific function by name
        result = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            # tools=tools,
            # tool_choice="structured_response",
            max_retries=max_retries
        )
        return result

llm_client = AzureLLMClient()

def google_search_fallback(query: str) -> list[str]:
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not google_api_key or not google_cx:
        return []

    for attempt in range(3):  # up to 3 retries
        try:
            params = {
                "key": google_api_key,
                "cx": google_cx,
                "q": query,
                "num": 5,
                "hl": "en",
                "safe": "active",
            }

            response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            if response.status_code == 429:
                wait = 2 ** attempt + 1
                time.sleep(wait)
                continue  # retry next loop
            response.raise_for_status()

            data = response.json()
            if "items" in data:
                return [f"{item['title']}: {item.get('snippet', '')}" for item in data["items"]]
            return []
        except Exception as e:
            time.sleep(2)
    return []

def web_search(query: str, num_results: int = 3, qid: str = "") -> str:
    """
    Multi-fallback search function:
    1. SerpAPI
    2. Tavily
    3. DuckDuckGo (free, no key)
    """
    results = google_search_fallback(query)
    if results:
        return " | ".join(results)

    # --- 1. Try SerpAPI ---
    # serpapi_key = os.getenv("SERPAPI_API_KEY")
    # if GoogleSearch and serpapi_key:
    #     try:
    #         params = {
    #             "api_key": serpapi_key,
    #             "engine": "google",
    #             "q": query,
    #             "num": num_results,
    #         }
    #         search = GoogleSearch(params)
    #         result = search.get_dict()
    #         snippets = []
    #         if "organic_results" in result:
    #             for item in result["organic_results"][:num_results]:
    #                 title = item.get("title", "")
    #                 snippet = item.get("snippet", "")
    #                 if snippet:
    #                     snippets.append(f"{title}: {snippet}")
    #         joined = " | ".join(snippets) if snippets else "No relevant evidence found"
    #         logging.info(f"[{qid}] SerpAPI success: {joined}")
    #         if joined != "No relevant evidence found":
    #             return joined
    #     except Exception as e:
    #         logging.warning(f"[{qid}] SerpAPI failed: {e}")

    # --- 2. Try Tavily ---
    tavily_key = os.getenv("TAVILY_API_KEY")
    if TavilyClient and tavily_key:
        try:
            tavily = TavilyClient(api_key=tavily_key)
            res = tavily.search(query, max_results=num_results)
            snippets = [doc["content"] for doc in res["results"][:num_results]]
            joined = " | ".join(snippets) if snippets else "No relevant evidence found"
            logging.info(f"[{qid}] Tavily success: {joined}")
            if joined != "No relevant evidence found":
                return joined
        except Exception as e:
            logging.warning(f"[{qid}] Tavily failed: {e}")

    # --- 3. Try DuckDuckGo ---
    if DDGS:
        try:
            with ddgs_lock:
                results = list(
                    ddgs_client.text(
                        query,
                        max_results=num_results,
                        region="us-en",   # restrict region & language to English
                        safesearch="moderate",
                    )
                )

            # Optional: filter again to drop non-English text if needed
            english_results = []
            for r in results[:num_results]:
                body = r.get("body", "")
                # keep only mostly-English snippets (heuristic)
                if sum(c.isascii() for c in body) / max(len(body), 1) > 0.8:
                    english_results.append(body)

            snippets = english_results or ["No relevant English evidence found"]
            joined = " | ".join(snippets)
            logging.info(f"[{qid}] DuckDuckGo (EN) success: {joined}")
            return joined

        except Exception as e:
            logging.warning(f"[{qid}] DuckDuckGo failed: {e}")

    # If all fail
    logging.error(f"[{qid}] All search providers failed for query: {query}")
    return "No relevant evidence found"



def safe_chat_completion(client, **kwargs):
    max_retries = 8
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait_time = min(2 ** attempt, 60)  # exponential backoff, max 60s
            logging.warning(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"❌ OpenAI API error: {e}")
            raise
    raise RuntimeError("❌ Exceeded max retries for OpenAI API")

# --- Reasoning Agent ---
class ReasoningAgent:
    def evaluate_with_llm(self, facts: Dict, evidence: Dict) -> Dict:
        qtype = facts["question_type"]
        options_text = "\n".join([f"{k}: {v}" for k, v in facts["options"].items()])
        evidence_text = "\n".join([f"{k}: {v}" for k, v in evidence.items()])

        if qtype in ["multi_choice", "open_ended_multi_choice"]:
            prompt = f"""
You are a clinical reasoning assistant.

Question: {facts['stem']}

Options:
{options_text}

Evidence from web search:
{evidence_text}

Task: Based on the evidence, return ONLY a valid JSON with two fields:
- selected_option: the letter of the correct option ("A","B","C","D")
- reasoning: 1–3 sentences explaining why (provide specific evidence from the search results).

No extra text, no markdown, no backticks.
"""
        elif qtype == "open_ended":
            prompt = f"""
You are a clinical reasoning assistant.

Question: {facts['stem']}

Evidence from web search:
{evidence_text}

Task: Provide ONLY a valid JSON with two fields:
- reasoning: 1–3 sentences answering the question concisely.
- selected_option: the letter ("A","B","C","D") that best matches your reasoning.

No extra text, no markdown, no backticks.
"""
        else:
            return {"selected_option": "", "reasoning": "Unsupported question type"}

        response = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4.1",
            temperature=0
        )
        logging.info(f"[{facts['id']}] LLM raw output: {response}")

        try:
            result = json.loads(response)
        except Exception:
            if qtype == "open_ended":
                result = {"reasoning": response}
            else:
                result = {"selected_option": "", "reasoning": response}

        return result

    def run(self, question_obj: dict) -> Dict:
        facts = {
            "id": question_obj.get("id", ""),
            "stem": question_obj.get("question", ""),
            "options": question_obj.get("options", {}),
            "question_type": question_obj.get("question_type", ""),
        }

        # gather evidence
        evidence = {}
        for opt_key, opt_val in facts["options"].items():
            query = f"{opt_val} {facts['stem']}"
            snippet = web_search(query, num_results=3, qid=facts["id"])
            evidence[opt_key] = snippet

        # evaluate
        llm_result = self.evaluate_with_llm(facts, evidence)
        qtype = facts["question_type"]

        if qtype in ["multi_choice", "open_ended_multi_choice"]:
            selected = llm_result.get("selected_option", "")
            final = {
                "id": facts["id"],
                "prediction": selected,
                "reasoning": llm_result.get("reasoning", ""),
                "choice": selected,  # same as prediction (A/B/C/D)
            }
        elif qtype == "open_ended":
            reasoning = llm_result.get("reasoning", "")
            # ép luôn choice = A/B/C/D (giống multi_choice)
            selected = llm_result.get("selected_option", "")
            final = {
                "id": facts["id"],
                "prediction": reasoning,
                "reasoning": reasoning,
                "choice": selected if selected else "A",  # fallback "A" nếu không có
            }
        else:
            final = {
                "id": facts["id"],
                "prediction": "",
                "reasoning": "Unsupported question type",
                "choice": "",
            }

        logging.info(f"[{facts['id']}] Final structured result: {final}")
        return final


# --- Batch Processor ---
def process_questions(questions: List[dict], max_workers: int = 5) -> List[dict]:
    agent = ReasoningAgent()
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(agent.run, q) for q in questions]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logging.error(f"❌ Failed to process question: {e}")
    return results


if __name__ == "__main__":
    input_file = "curebench_testset_phase1.jsonl"
    questions = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
    questions = questions[2041:]  # chỉ dùng 2079 câu đầu tiên
    # thử tất cả câu (2079) song song
    rows = []
    batch_size = 5
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i+batch_size]
        answers = process_questions(batch, max_workers=20)
        with open("agent_output_results_16.jsonl", "a", encoding="utf-8") as f:
            for ans in answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
                rows.append({
                    "id": ans.get("id", ""),
                    "prediction": ans.get("prediction", ""),
                    "reasoning_trace": ans.get("reasoning", ""),
                    "choice": ans.get("choice", "")
                })
    import pandas as pd
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=["id", "prediction", "reasoning_trace", "choice"])

    # Save to CSV
    df.to_csv("/home/tramnguyen/explo/curebench-task2/agent_output_results_16.csv", index=False, encoding="utf-8")
    # Save CSV
    print("✅ Done! Results saved to agent_output_results_16.jsonl, agent_output_results_16.csv, and detailed logs in agent_run.log")