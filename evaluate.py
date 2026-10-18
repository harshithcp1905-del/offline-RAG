import pandas as pd
import requests
import time

# Evaluation dataset with multiple acceptable keywords per question
eval_dataset = [
    {
        "question": "What is FastAPI based on?",
        "acceptable_keywords": ["python", "type hints", "starlette"]
    },
    {
        "question": "What kind of engine is Qdrant?",
        "acceptable_keywords": ["vector", "similarity", "search"]
    },
    {
        "question": "What is RAG?",
        "acceptable_keywords": ["retrieval", "augmented", "generation"]
    },
    {
        "question": "What framework is Sentence-Transformers built on?",
        "acceptable_keywords": ["pytorch", "torch"]
    }
]

def run_evaluation():
    results = []
    endpoint = "http://localhost:8001/query"

    print("Running evaluation suite...")
    for item in eval_dataset:
        print(f"-> Testing: '{item['question']}'")
        start_time = time.time()
        try:
            res = requests.post(endpoint, json={"query": item["question"]}, timeout=120)
            latency = time.time() - start_time
            
            if res.status_code == 200:
                data = res.json()
                answer = data["answer"].lower()
                
                # Check if ANY of the acceptable keywords are in the answer
                passed = any(kw in answer for kw in item["acceptable_keywords"])
                
                results.append({
                    "question": item["question"],
                    "latency_sec": round(latency, 3),
                    "passed": passed,
                    "answer": data["answer"]
                })
                print(f"   [PASS: {passed}] ({latency:.3f}s)")
            else:
                results.append({
                    "question": item["question"],
                    "latency_sec": round(latency, 3),
                    "passed": False,
                    "answer": "ERROR"
                })
                print(f"   [FAIL: Server Error Status {res.status_code}]")
        except Exception as e:
            print(f"Failed to connect to backend: {e}")
            break

    if results:
        df = pd.DataFrame(results)
        df.to_csv("eval_baseline_results.csv", index=False)
        print("\n==============================")
        print("Evaluation complete! Results saved to eval_baseline_results.csv")
        print(f"Pass Rate: {df['passed'].mean() * 100}% | Average Latency: {df['latency_sec'].mean():.3f}s")
        print("==============================")

if __name__ == "__main__":
    run_evaluation()