import requests, sys

BASE_URL = "http://localhost:8000/ask"

CASES = [
  # question, must_contain (substring in answer), expect_sources>0?
  ("What programming languages does AcmeTech support?", "JavaScript, TypeScript, Python, and Go", True),
  ("How does authentication work?", "JWT", True),
  ("How does AcmeTech ensure data security?", "AES-256", True),
  ("Does AcmeTech offer a free plan?", "1,000 API requests", True),
  ("What databases are supported?", "PostgreSQL", True),
  ("Can AcmeTech scale with my application?", "Kubernetes", True),
  ("Do you provide AI capabilities?", "RAG", True),
  ("What support plans are available?", "Enterprise", True),
  ("How can I get started?", "API key", True),
  ("Do you offer on-prem pricing details?", "Not specified in the docs.", False),
]

def run():
  ok = 0
  for q, must, expect_src in CASES:
    r = requests.post(BASE_URL, json={"question": q}, timeout=20)
    r.raise_for_status()
    data = r.json()
    ans = data.get("answer","")
    srcs = data.get("sources", [])
    passed = (must in ans) and ((len(srcs)>0) == expect_src)
    print(f"[{'OK' if passed else 'FAIL'}] {q}\n  answer: {ans}\n  sources: {len(srcs)}; snippet: {srcs[0]['text'][:80] if srcs else '-'}\n")
    ok += int(passed)
  print(f"Passed {ok}/{len(CASES)}")
  if ok != len(CASES):
    sys.exit(1)

if __name__ == "__main__":
  run()

