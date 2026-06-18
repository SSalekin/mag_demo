#!/usr/bin/env python3
"""Benchmark for Titan External Memory Agent."""

from __future__ import annotations

import argparse, time
from pathlib import Path
from typing import Dict, List, Tuple
import torch

from titan_implementation import TitanExternalMemory, ask_with_memory

TestCase = Dict[str, object]

TESTS: List[TestCase] = [
    {"name":"simple_location","store":["Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam."],"question":"Where is Sarah Nguyen from?","expected_any":["hanoi"],"forbidden":["danang","da nang","rennes","lyon"]},
    {"name":"atomic_profile_retains_study_after_location_update","store":["Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam. She studies natural language processing and works on a sentiment analysis project.","Actually, Sarah Nguyen is now from Da Nang, Vietnam."],"question":"What does Sarah Nguyen study?","expected_any":["natural language processing","nlp"],"forbidden":[]},
    {"name":"location_update","store":["Clara Rossi is a 20-year-old design student from Milan, Italy.","Actually, Clara Rossi is now from Rome, Italy."],"question":"Where is Clara Rossi from?","expected_any":["rome"],"forbidden":["milan"]},
    {"name":"favorite_color_update","store":["Sarah's favorite color is yellow.","Actually, Sarah's favorite color is now black."],"question":"What is Sarah's favorite color?","expected_any":["black"],"forbidden":["yellow"]},
    {"name":"favorite_language_update","store":["Nathan's favorite programming language is Scala.","Actually, Nathan's favorite programming language is now Python."],"question":"What is Nathan's favorite programming language?","expected_any":["python"],"forbidden":["scala"]},
    {"name":"secret_code_update","store":["Lucas's secret code is 8392.","Actually, Lucas's secret code is now 4917."],"question":"What is Lucas's secret code?","expected_any":["4917"],"forbidden":["8392"]},
    {"name":"multi_profile_separation","store":["Emma Leclerc is a 21-year-old student from Nantes, France. She studies computer science and loves robotics.","Maya Dubois is a 23-year-old woman from Marseille, France. She speaks French, English, and Spanish.","Kenji Tanaka is a 26-year-old software engineer from Osaka, Japan."],"question":"Which languages does Maya speak?","expected_any":["french","english","spanish"],"forbidden":["osaka","robotics","nantes"]},
    {"name":"forget_target_only","store":["Adam Wilson is a 29-year-old machine learning engineer from Manchester.","Adam works on recommendation systems.","Adam's temporary access code is AX-9021."],"forget":"Adam temporary access code","question":"What does Adam work on?","expected_any":["recommendation"],"forbidden":["ax-9021"]},
    {"name":"forget_secret_code","store":["Lina's secret code is LA-2026."],"forget":"Lina secret code","question":"What is Lina's secret code?","expected_any":["do not know","don't know","not know","no memory"],"forbidden":["la-2026"]},
    {"name":"distractor_resistance","store":["Sofia Garcia is interested in radiology, medical imaging, and AI-assisted diagnosis.","Sofia's research group is called MedVision Lab.","The weather in Da Nang is very sunny today.","I ate chicken rice for lunch.","The hotel lobby has a large blue sofa.","There are many motorbikes near the city center."],"question":"What is Sofia's research group called?","expected_any":["medvision"],"forbidden":["blue sofa","chicken rice","weather"]},
]

def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())

def answer_passes(answer: str, expected_any: List[str], forbidden: List[str]) -> Tuple[bool,str]:
    lower=normalize(answer)
    no_memory_expected=any(normalize(k) in {"do not know","don't know","not know","no memory"} for k in expected_any)
    if not lower and no_memory_expected:
        return True, "ok (no retrieved memory)"
    expected_ok=any(normalize(k) in lower for k in expected_any)
    forbidden_hit=[w for w in forbidden if normalize(w) in lower]
    if not expected_ok: return False, f"missing expected keyword(s): {expected_any}"
    if forbidden_hit: return False, f"contains forbidden keyword(s): {forbidden_hit}"
    return True, "ok"

def retrieval_only_answer(memory: TitanExternalMemory, question: str, topk: int) -> str:
    results=memory.retrieve(question, k=topk, min_score=0.12)
    return "\n".join(item.text for _,_,item in results)

def run_test(test: TestCase, args: argparse.Namespace) -> Dict[str,object]:
    memory_file=Path(args.memory_file)
    if memory_file.exists(): memory_file.unlink()
    memory=TitanExternalMemory(memory_path=memory_file, d_model=args.d_model, hidden_dim=args.hidden_dim, max_items=args.max_items, device=args.device)
    memory.reset()
    for statement in test.get("store", []):
        memory.store_text(str(statement))
    if "forget" in test:
        candidates=memory.search_forget_candidates(str(test["forget"]), k=1)
        if candidates:
            memory.deactivate_ids([candidates[0][2].id], reason=f"benchmark forget: {test['forget']}")
    start=time.perf_counter()
    if args.retrieval_only:
        answer=retrieval_only_answer(memory, str(test["question"]), args.topk)
    else:
        answer=ask_with_memory(str(test["question"]), memory, args.ollama_model, args.topk, args.min_score, args.debug, False)
    latency=time.perf_counter()-start
    passed,reason=answer_passes(answer, list(test["expected_any"]), list(test.get("forbidden", [])))
    return {"name":test["name"],"passed":passed,"reason":reason,"latency":latency,"answer":answer,"active_memories":len(memory.active_items),"inactive_memories":len(memory.inactive_items)}

def main() -> int:
    base_dir=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description="Benchmark Titan external memory.")
    parser.add_argument("--ollama-model", default="gemma2:2b")
    parser.add_argument("--memory-file", default=str(base_dir/"benchmark_titan_memory_store.pt"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.12)
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu","cuda"])
    args=parser.parse_args()
    results=[]
    print("\nRunning Titan memory benchmark...\n")
    for i,test in enumerate(TESTS,1):
        print(f"[{i}/{len(TESTS)}] {test['name']} ...")
        try:
            result=run_test(test,args)
        except Exception as exc:
            result={"name":test["name"],"passed":False,"reason":f"exception: {exc}","latency":0.0,"answer":"","active_memories":0,"inactive_memories":0}
        results.append(result)
        status="PASS" if result["passed"] else "FAIL"
        print(f"  {status} | {result['reason']} | latency={result['latency']:.2f}s")
        if not result["passed"] or args.debug:
            print(f"  Answer: {result['answer']}")
        print()
    passed=sum(1 for r in results if r["passed"]); total=len(results)
    avg=sum(float(r["latency"]) for r in results)/max(1,total)
    print("="*72); print(f"Score: {passed}/{total} ({passed/total*100:.1f}%)"); print(f"Average latency: {avg:.2f}s"); print("="*72)
    print("\nDetailed results:")
    for r in results:
        print(f"- {'PASS' if r['passed'] else 'FAIL'} | {r['name']} | {r['reason']}")
    return 0 if passed==total else 1

if __name__ == "__main__":
    raise SystemExit(main())
