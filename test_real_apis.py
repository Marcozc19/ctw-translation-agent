"""
Real end-to-end test against live APIs.
Loads keys from backend/.env, runs a 5-row translation job through all 4 agents.
"""
import asyncio, json, os, re, sys, time, io
from pathlib import Path

# Load .env manually
env_path = Path(__file__).parent / "backend" / ".env"
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

import pandas as pd

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
BLUE  = "\033[94m"
YELLOW= "\033[93m"
DIM   = "\033[90m"
RESET = "\033[0m"

def ok(label, detail=""): print(f"  {GREEN}✓{RESET}  {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
def fail(label, detail=""): print(f"  {RED}✗{RESET}  {label}" + (f"  {DIM}{detail}{RESET}" if detail else "")); sys.exit(1)
def info(msg): print(f"  {BLUE}•{RESET}  {msg}")
def section(t): print(f"\n{BOLD}{'─'*60}\n  {t}\n{'─'*60}{RESET}")

# ── Test CSV — 5 rows, varied content ────────────────────────────────────────
SAMPLE = """id,product,description_zh,price
1,Wireless Earbuds,高品质无线蓝牙耳机，降噪效果出色，续航20小时,79.99
2,Coffee Maker,全自动咖啡机，支持多种冲泡模式，操作简便,149.99
3,Running Shoes,专业马拉松跑鞋，轻量碳纤维底，回弹性能卓越,239.99
4,Face Serum,含玻尿酸精华液，深层补水保湿，适合干燥肌肤,49.99
5,Smart Watch,智能手表，心率监测，睡眠分析，防水50米,199.99"""

df = pd.read_csv(io.StringIO(SAMPLE))

def _parse_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

# ── Agent 1: Column Detection ─────────────────────────────────────────────────
section("Agent 1 — Column Detection (rule-based)")

CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
detected = [c for c in df.columns
            if sum(1 for t in df[c].dropna().astype(str).head(10) if CJK_RE.search(t)) / max(len(df[c].dropna()), 1) >= 0.3]
ok(f"Detected Chinese columns: {detected}")

# ── Agent 2: DeepSeek V3 ──────────────────────────────────────────────────────
section("Agent 2 — DeepSeek V3 (low-cost translator)")

from openai import AsyncOpenAI
ds = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

batch = [{"id": int(row.id), "description_zh": row.description_zh} for _, row in df.iterrows()]
out_keys = ["description_zh_en", "description_zh_es"]

ds_system = "You are a professional translator specialising in Chinese. Output ONLY a valid JSON array — no commentary, no markdown."
ds_user = (
    f"Translate the 'description_zh' field to English and Spanish.\n"
    f"Return a JSON array. Each object must have 'id' (unchanged) plus: {json.dumps(out_keys)}.\n"
    f"Input rows:\n{json.dumps(batch)}"
)

async def run_deepseek():
    t = time.time()
    info("Calling DeepSeek V3...")
    resp = await ds.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": ds_system}, {"role": "user", "content": ds_user}],
        temperature=0.1, max_tokens=2048,
    )
    elapsed = time.time() - t
    raw = resp.choices[0].message.content
    parsed = _parse_json(raw)
    if not parsed:
        fail("JSON parse failed", raw[:200])
    ok(f"Got {len(parsed)} translations in {elapsed:.1f}s  (model: {resp.model})")
    for item in parsed:
        info(f"  id={item['id']}  EN: {item.get('description_zh_en','')[:55]}...")
    return parsed

ds_results = asyncio.run(run_deepseek())

# ── Agent 3: Gemini back-translation + similarity ─────────────────────────────
section("Agent 3 — Gemini 2.0 Flash (evaluator)")

from google import genai as gai
gemini = gai.Client(api_key=os.environ["GOOGLE_API_KEY"])
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# We'll do simple string-overlap similarity without sentence-transformers
def simple_sim(a: str, b: str) -> float:
    """Token overlap as a lightweight similarity proxy (no torch needed)."""
    a_tokens = set(re.findall(r'[一-鿿]|[a-z]+', a.lower()))
    b_tokens = set(re.findall(r'[一-鿿]|[a-z]+', b.lower()))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))

async def run_gemini():
    scores = []
    for item in ds_results:
        original_zh = next(r.description_zh for _, r in df.iterrows() if int(r.id) == item["id"])
        en_trans = item.get("description_zh_en", "")

        t = time.time()
        prompt = (
            f"Translate the following English text back to Simplified Chinese. "
            f"Return ONLY the Chinese translation, nothing else.\n\nText: {en_trans}"
        )
        response = await gemini.aio.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        back = response.text.strip()
        elapsed = time.time() - t

        score = simple_sim(original_zh, back)
        verdict = "high" if score >= 0.40 else "low"   # token-overlap scale ≠ embedding scale
        info(f"  id={item['id']}  score={score:.2f} ({verdict})  back: {back[:40]}...  [{elapsed:.1f}s]")
        scores.append((item["id"], score, verdict))

    ok(f"Evaluated {len(scores)} rows — avg score {sum(s for _,s,_ in scores)/len(scores):.2f}")
    return scores

eval_scores = asyncio.run(run_gemini())

# ── Agent 4: Claude Haiku (high-cost fallback) ────────────────────────────────
section("Agent 4 — Claude Haiku 4.5 (high-quality fallback)")

import anthropic
claude = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")

# Escalate the two lowest-scoring rows to Haiku
escalate_ids = sorted(eval_scores, key=lambda x: x[1])[:2]
escalate_batch = [{"id": int(r.id), "description_zh": r.description_zh} for _, r in df.iterrows()
                  if int(r.id) in {e[0] for e in escalate_ids}]

haiku_system = "You are an expert translator. Return ONLY a valid JSON array — no commentary, no markdown."
haiku_user = (
    f"These rows need higher-quality translation to English and Spanish.\n"
    f"Output keys per object: 'id' + {json.dumps(out_keys)}.\n"
    f"Rows:\n{json.dumps(escalate_batch)}"
)

async def run_haiku():
    t = time.time()
    info(f"Escalating {len(escalate_batch)} rows to Haiku (ids: {[r['id'] for r in escalate_batch]})...")
    resp = await claude.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        system=haiku_system,
        messages=[{"role": "user", "content": haiku_user}],
    )
    elapsed = time.time() - t
    raw = resp.content[0].text
    parsed = _parse_json(raw)
    if not parsed:
        fail("Haiku JSON parse failed", raw[:200])
    ok(f"Got {len(parsed)} Haiku translations in {elapsed:.1f}s  (model: {HAIKU_MODEL})")
    for item in parsed:
        info(f"  id={item['id']}  EN: {item.get('description_zh_en','')[:55]}...")
    return parsed

haiku_results = asyncio.run(run_haiku())

# ── Assemble final output ──────────────────────────────────────────────────────
section("Final Output CSV")

haiku_map = {item["id"]: item for item in haiku_results}
en_col, es_col, conf_col = [], [], []

for _, row in df.iterrows():
    rid = int(row.id)
    src = next((r for r in ds_results if r["id"] == rid), {})
    haiku = haiku_map.get(rid)
    final = haiku if haiku else src
    _, score, verdict = next(s for s in eval_scores if s[0] == rid)
    en_col.append(final.get("description_zh_en", ""))
    es_col.append(final.get("description_zh_es", ""))
    conf_col.append("high" if verdict == "high" else "review" if rid in haiku_map else "low")

out_df = df.copy()
out_df["description_zh_en"] = en_col
out_df["description_zh_es"] = es_col
out_df["translation_confidence"] = conf_col

print()
for _, row in out_df.iterrows():
    flag = "  ⚠ review" if row.translation_confidence == "review" else ""
    print(f"  [{row.translation_confidence:>6}]  ZH: {row.description_zh[:35]:<35}")
    print(f"            EN: {row.description_zh_en[:65]}")
    print(f"            ES: {row.description_zh_es[:65]}{flag}")
    print()

out_path = Path(__file__).parent / "test_real_output.csv"
out_df.to_csv(out_path, index=False)
ok(f"Saved → {out_path}")
print(f"\n  {BOLD}All 4 agents working with live APIs.{RESET}\n")
