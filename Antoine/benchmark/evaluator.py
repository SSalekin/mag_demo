#!/usr/bin/env python3
"""
Benchmark for Antoine Models (Transformer, LSTM, GRU)
=====================================================

Run from the Antoine root directory:
    python -m benchmark.evaluator
"""

from __future__ import annotations

import argparse
import time
import sys
import os
from typing import Dict, List, Tuple

# Ensure we can import from models/ if running directly from benchmark folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.transformer_model import TransformerModel
from models.lstm_model import LstmModel
from models.gru_model import GruModel
from models.gnn_model import GnnModel

TestCase = Dict[str, object]

TESTS: List[TestCase] = [
    {
        "name": "simple_location",
        "store": ["Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam."],
        "question": "Where is Sarah Nguyen from?",
        "expected_any": ["hanoi"],
        "forbidden": ["danang", "da nang", "rennes", "lyon"],
    },
    {
        "name": "atomic_profile_retains_study_after_location_update",
        "store": [
            "Sarah Nguyen is a 23-year-old AI student from Hanoi, Vietnam. She studies natural language processing and works on a sentiment analysis project.",
            "Actually, Sarah Nguyen is now from Da Nang, Vietnam.",
        ],
        "question": "What does Sarah Nguyen study?",
        "expected_any": ["natural language processing", "nlp"],
        "forbidden": [],
    },
    {
        "name": "location_update",
        "store": [
            "Clara Rossi is a 20-year-old design student from Milan, Italy.",
            "Actually, Clara Rossi is now from Rome, Italy.",
        ],
        "question": "Where is Clara Rossi from?",
        "expected_any": ["rome"],
        "forbidden": ["milan"],
    },
    {
        "name": "favorite_color_update",
        "store": [
            "Sarah's favorite color is yellow.",
            "Actually, Sarah's favorite color is now black.",
        ],
        "question": "What is Sarah's favorite color?",
        "expected_any": ["black"],
        "forbidden": ["yellow"],
    },
    {
        "name": "favorite_language_update",
        "store": [
            "Nathan's favorite programming language is Scala.",
            "Actually, Nathan's favorite programming language is now Python.",
        ],
        "question": "What is Nathan's favorite programming language?",
        "expected_any": ["python"],
        "forbidden": ["scala"],
    },
    {
        "name": "secret_code_update",
        "store": [
            "Lucas's secret code is 8392.",
            "Actually, Lucas's secret code is now 4917.",
        ],
        "question": "What is Lucas's secret code?",
        "expected_any": ["4917"],
        "forbidden": ["8392"],
    },
    {
        "name": "multi_profile_separation",
        "store": [
            "Emma Leclerc is a 21-year-old student from Nantes, France. She studies computer science and loves robotics.",
            "Maya Dubois is a 23-year-old woman from Marseille, France. She speaks French, English, and Spanish.",
            "Kenji Tanaka is a 26-year-old software engineer from Osaka, Japan.",
        ],
        "question": "Which languages does Maya speak?",
        "expected_any": ["french", "english", "spanish"],
        "forbidden": ["osaka", "robotics", "nantes"],
    },
    {
        "name": "forget_target_only",
        "store": [
            "Adam Wilson is a 29-year-old machine learning engineer from Manchester.",
            "Adam works on recommendation systems.",
            "Adam's temporary access code is AX-9021.",
        ],
        "forget": "Adam temporary access code",
        "question": "What does Adam work on?",
        "expected_any": ["recommendation"],
        "forbidden": ["ax-9021"],
    },
    {
        "name": "forget_secret_code",
        "store": ["Lina's secret code is LA-2026."],
        "forget": "Lina secret code",
        "question": "What is Lina's secret code?",
        "expected_any": ["do not know", "don't know", "not know", "no memory"],
        "forbidden": ["la-2026"],
    },
    {
        "name": "distractor_resistance",
        "store": [
            "Sofia Garcia is interested in radiology, medical imaging, and AI-assisted diagnosis.",
            "Sofia's research group is called MedVision Lab.",
            "The weather in Da Nang is very sunny today.",
            "I ate chicken rice for lunch.",
            "The hotel lobby has a large blue sofa.",
            "There are many motorbikes near the city center.",
        ],
        "question": "What is Sofia's research group called?",
        "expected_any": ["medvision"],
        "forbidden": ["blue sofa", "chicken rice", "weather"],
    },
]


def normalize(text: str) -> str:
    return " ".join(str(text).lower().split())


def answer_passes(answer: str, expected_any: List[str], forbidden: List[str]) -> Tuple[bool, str]:
    lower = normalize(answer)

    no_memory_expected = any(
        normalize(keyword) in {"do not know", "don't know", "not know", "no memory"}
        for keyword in expected_any
    )
    if not lower and no_memory_expected:
        return True, "ok (no retrieved memory)"

    expected_ok = any(normalize(keyword) in lower for keyword in expected_any)
    forbidden_hit = [word for word in forbidden if normalize(word) in lower]
    if not expected_ok:
        return False, f"missing expected keyword(s): {expected_any}"
    if forbidden_hit:
        return False, f"contains forbidden keyword(s): {forbidden_hit}"
    return True, "ok"


def run_test(test: TestCase, model_class, args: argparse.Namespace) -> Dict[str, object]:
    model = model_class(model_name=args.ollama_model, max_capacity=args.capacity)
    model.clear_memory()

    # Load memory by simulating a conversation
    for statement in test.get("store", []):
        model.add_user_message(f"Please remember this information: {statement}")
        # Add a simulated acknowledgement so the model context treats it nicely
        model.add_assistant_message("Understood. I will remember that.")

    if "forget" in test:
        # The Antoine models do not have explicit targeted forgetting. 
        # We simulate a "forget" command by asking the user to forget, 
        # but the models' actual architectures (sliding window/LSTM) might not natively forget it 
        # unless it falls out of the context window or decays.
        model.add_user_message(f"Please forget this: {test['forget']}")
        model.add_assistant_message("Acknowledged. I have forgotten it.")

    start = time.perf_counter()
    
    # Ask the question
    model.add_user_message(str(test["question"]))
    
    # Generate the answer stream and collect it
    answer_chunks = []
    stream = model.generate_response_stream()
    for chunk in stream:
        answer_chunks.append(chunk)
    
    answer = "".join(answer_chunks)
    
    # Add the assistant message to memory to complete the interaction
    model.add_assistant_message(answer)
        
    latency = time.perf_counter() - start

    passed, reason = answer_passes(
        answer,
        expected_any=list(test["expected_any"]),
        forbidden=list(test.get("forbidden", [])),
    )

    return {
        "name": test["name"],
        "passed": passed,
        "reason": reason,
        "latency": latency,
        "answer": answer,
        "active_memories": model.get_active_tokens_count(),
        "dropped_tokens": model.get_dropped_tokens_count()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Antoine memory architectures.")
    parser.add_argument("--arch", choices=["transformer", "lstm", "gru", "gnn"], default=None, help="Which memory architecture to test")
    parser.add_argument("--ollama-model", default="llama3", help="The Ollama model to use for inference")
    parser.add_argument("--capacity", type=int, default=8192, help="The maximum memory capacity")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    if args.arch is None:
        print("\n--- Architecture Selection ---")
        print("1. Transformer (KV-Cache)")
        print("2. LSTM (Hidden State)")
        print("3. GRU (Hidden State)")
        print("4. GNN (Graph Neural Network)")
        while True:
            choice = input("Select an architecture to benchmark (1, 2, 3 or 4): ").strip()
            if choice == '1':
                args.arch = "transformer"
                break
            elif choice == '2':
                args.arch = "lstm"
                break
            elif choice == '3':
                args.arch = "gru"
                break
            elif choice == '4':
                args.arch = "gnn"
                break
            print("Invalid choice. Please enter 1, 2, 3 or 4.")

    arch_map = {
        "transformer": TransformerModel,
        "lstm": LstmModel,
        "gru": GruModel,
        "gnn": GnnModel
    }
    model_class = arch_map[args.arch]

    results = []
    print(f"\nRunning Antoine {args.arch.upper()} memory benchmark (Model: {args.ollama_model})...\n")

    for i, test in enumerate(TESTS, start=1):
        print(f"[{i}/{len(TESTS)}] {test['name']} ...")
        try:
            result = run_test(test, model_class, args)
        except Exception as exc:
            result = {
                "name": test["name"],
                "passed": False,
                "reason": f"exception: {exc}",
                "latency": 0.0,
                "answer": "",
                "active_memories": 0,
                "dropped_tokens": 0
            }

        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  {status} | {result['reason']} | latency={result['latency']:.2f}s | active_tokens={result['active_memories']} | dropped_tokens={result['dropped_tokens']}")
        if not result["passed"] or args.debug:
            print(f"  Answer: {result['answer']}")
        print()

    passed = sum(1 for row in results if row["passed"])
    total = len(results)
    avg_latency = sum(float(row["latency"]) for row in results) / max(1, total)

    print("=" * 72)
    print(f"Score ({args.arch.upper()}): {passed}/{total} ({passed / total * 100:.1f}%)")
    print(f"Average latency: {avg_latency:.2f}s")
    print("=" * 72)

    print("\nDetailed results:")
    for row in results:
        status = "PASS" if row["passed"] else "FAIL"
        print(f"- {status} | {row['name']} | {row['reason']}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
