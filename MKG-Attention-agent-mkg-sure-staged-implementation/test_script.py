import hashlib
import json
import re
import time
from http import client
from random import randint
 
# =========================
# INPUT / OUTPUT PATHS
# =========================
INPUT_PATH = ""
OUTPUT_TXT_PATH = ""
 
# =========================
# USER AGENTS
# =========================
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3)...",
]
 
# =========================
# GPT API CALL
# =========================
def query_gpt(prompt: str, model_name: str = "gpt-5.1-thinking-all") -> str:
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
        "Authorization": 'sk-BEaiDaXxebrLPaOUuq68nm6rZQGzVjgxfYKxilUDkt6JAMm2',  # prefer env var
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
        print(f"❌ API error: {exc}")
        return ""
    finally:
        try:
            conn.close()
        except Exception:
            pass
 
 
# =========================
# NORMALIZER
# =========================
def normalize_triplet_output(response: str) -> list[str]:
    triplets: list[str] = []
 
    for line in response.splitlines():
        line = re.sub(r"^\d+\.\s*", "", line.strip())
 
        # Convert "A | B | C" → "(A, B, C)"
        if "|" in line and not line.startswith("("):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 3:
                line = f"({parts[0]}, {parts[1]}, {parts[2]})"
 
        if line.startswith("(") and line.endswith(")"):
            triplets.append(line)
 
    return triplets
 
 
# =========================
# PROMPTS
# =========================
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
 
 
# =========================
# MAIN
# =========================
def main() -> None:
    seen: set[str] = set()
    prompt_count = 1
 
    # Open output in append mode and flush after each prompt
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, open(
        OUTPUT_TXT_PATH, "a", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            data = json.loads(line)
 
            qid = data.get("qid", f"line_{line_no}")
            question = (data.get("question") or "").strip()
            if not question:
                continue
 
            items: list[tuple[str, str, str]] = []
 
            # ---- ANSWERS ----
            for ans in data.get("answers", []):
                modality = ans.get("modality")
 
                if modality == "image":
                    for inst in ans.get("image_instances", []):
                        caption = inst.get("image_caption")
                        doc_id = inst.get("doc_id", "")
                        if isinstance(caption, str) and caption.strip():
                            items.append((caption, "image", doc_id))
 
                if modality == "text":
                    for inst in ans.get("text_instances", []):
                        txt = inst.get("full_text") or inst.get("text")
                        doc_id = inst.get("doc_id", "")
                        if isinstance(txt, str) and txt.strip():
                            items.append((txt, "text", doc_id))
 
            # ---- SUPPORTING CONTEXT ----
            for ctx in data.get("supporting_context", []):
                doc_id = ctx.get("doc_id", "")
 
                caption = ctx.get("image_caption")
                if isinstance(caption, str) and caption.strip():
                    items.append((caption, "image", doc_id))
 
                # If your supporting_context includes text/full_text, include it too:
                txt = ctx.get("full_text") or ctx.get("text")
                if isinstance(txt, str) and txt.strip():
                    items.append((txt, "text", doc_id))
 
            # ---- EXTRACT ----
            for evidence, kind, doc_id in items:
                hash_key = hashlib.md5((question + evidence).encode()).hexdigest()
                if hash_key in seen:
                    continue
                seen.add(hash_key)
 
                prompt = (
                    prompt_for_image(question, evidence)
                    if kind == "image"
                    else prompt_for_text(question, evidence)
                )
 
                print(f" Prompt {prompt_count} ({kind}) sending | qid={qid} doc_id={doc_id}")
 
                response = query_gpt(prompt)
                triplets = normalize_triplet_output(response)
 
                # Write results for THIS prompt immediately, then flush to disk
                if triplets:
                    for t in triplets:
                        fout.write(t + "\n")
                else:
                    fout.write("(No triplets found)\n")
 
                fout.flush()
 
                print(
                    f" Prompt {prompt_count} ({kind}) done | triplets={len(triplets)} | qid={qid} doc_id={doc_id}"
                )
 
                prompt_count += 1
                time.sleep(1)
 
    print("DONE. Triplets saved to:", OUTPUT_TXT_PATH)
 
 
if __name__ == "__main__":
    main()