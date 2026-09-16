from dotenv import load_dotenv
load_dotenv()

import json
import re
import uuid
import requests
import torch
import warnings
import logging
from urllib.parse import quote

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_tavily import TavilySearch
from langchain.tools import tool

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Both `max_new_tokens`.*and `max_length`.*")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
pipe = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=False,
    return_full_text=False,
)

@tool
def get_weather(city: str) -> dict:
    """Get the current weather of a city."""
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={quote(city)}&count=1&language=en&format=json"
    )
    geo_data = requests.get(geo_url, timeout=20).json()
    if "results" not in geo_data:
        return {"error": f"City '{city}' not found"}
    loc = geo_data["results"][0]
    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['latitude']}&longitude={loc['longitude']}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m"
    )
    current = requests.get(weather_url, timeout=20).json()["current"]
    return {
        "city": loc["name"],
        "country": loc.get("country"),
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "feels_like": current["apparent_temperature"],
        "wind_speed": current["wind_speed_10m"],
    }

search_tools = TavilySearch(max_results=5)

@tool
def get_news(city: str) -> str:
    """Get the latest news about a city."""
    result = search_tools.invoke({"query": f"latest news in {city}"})
    return json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result

TOOLS = [get_weather, get_news]
TOOL_MAP = {t.name: t for t in TOOLS}

# OpenAI-style schemas so Qwen's chat template can render them
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.args_schema.model_json_schema(),
        },
    }
    for t in TOOLS
]

SYSTEM = """You are a city information assistant.
Use get_weather for weather/temperature/humidity/wind.
Use get_news for latest or current news about a city.
Never invent weather numbers or news headlines.
Never write Python like getattr() or fake article lists.
If you need real-time info, emit a tool call."""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        args = data.get("arguments") or data.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"city": args}
        calls.append({
            "name": data["name"],
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
        })
    return calls

def generate(messages: list[dict]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=OPENAI_TOOLS,
        add_generation_prompt=True,
        tokenize=False,
    )
    return pipe(prompt)[0]["generated_text"].strip()

def run_agent(user_input: str, max_rounds: int = 5) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_input},
    ]
    for _ in range(max_rounds):
        text = generate(messages)
        calls = parse_tool_calls(text)
        if not calls:
            return text
        messages.append({"role": "assistant", "content": text})
        for call in calls:
            tool = TOOL_MAP.get(call["name"])
            if tool is None:
                payload = json.dumps({"error": f"unknown tool {call['name']}"})
            else:
                payload = tool.invoke(call["args"])
                if not isinstance(payload, str):
                    payload = json.dumps(payload, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "name": call["name"],
                "content": payload,
            })
    return "Stopped after too many tool rounds."

print("City Agent\nType exit to quit")
while True:
    user_input = input("You : ").strip()
    if user_input.lower() == "exit":
        break
    print("BOT:", run_agent(user_input))
