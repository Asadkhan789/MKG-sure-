import json
import os
import re
from http import client
from random import randint
from dotenv import load_dotenv

load_dotenv(".env", override=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3)...",
]


def query_gpt(prompt: str, model_name: str = "claude-sonnet-4-6-thinking") -> str:
    api_key = os.environ.get("AIGCBEST_API_KEY", "")
    if not api_key:
        print("❌ Set AIGCBEST_API_KEY (OpenAI-compatible key for api2.aigcbest.top)")
        return ""

    random_agent = USER_AGENTS[randint(0, len(USER_AGENTS) - 1)]
    conn = client.HTTPSConnection("api2.aigcbest.top")

    payload = json.dumps(
        {
            "model": model_name,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
    )

    headers = {
        "Accept": "application/json",
        "Authorization": api_key,
        "User-Agent": random_agent,
        "Content-Type": "application/json",
    }

    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        response = res.read()
        data = json.loads(response.decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"❌ API error: {exc} | {response}")
        return ""
    finally:
        try:
            conn.close()
        except Exception:
            pass


def normalize_triplet_output(response: str) -> list[str]:
    triplets: list[str] = []

    for line in response.splitlines():
        line = re.sub(r"^\d+\.\s*", "", line.strip())

        if "|" in line and not line.startswith("("):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                line = f"({parts[0]}, {parts[1]}, {parts[2]})"

        if line.startswith("(") and line.endswith(")"):
            triplets.append(line)

    return triplets


def prompt_for_image(question: str, caption: str) -> str:
    return f"""
Extract ONLY visual knowledge triplets that help answer the question.

STRICT FORMAT: (subject, relation, object)

Rules:
- Use ONLY what is present in the image caption (visual cues, layout, symbols, colors, shapes)

Question:
{question}

Image Caption:
\"\"\"{caption}\"\"\"

Triplets:
""".strip()


def prompt_for_text(question: str, full_text: str) -> str:
    return f"""
Extract ONLY factual knowledge triplets that help answer the question.

STRICT FORMAT: (subject, relation, object)

Rules:
- Use ONLY information present in the text

Question:
{question}

Text:
\"\"\"{full_text}\"\"\"

Triplets:
""".strip()


def main() -> None:
    prompt = prompt_for_text(
        question="What color is the door?",
        full_text="The house has a red front door and two windows.",
    )
    raw = query_gpt(prompt)
    if not raw:
        return
    print(raw)
    triplets = normalize_triplet_output(raw)
    print("Parsed triplets:", triplets)


if __name__ == "__main__":
    main()
