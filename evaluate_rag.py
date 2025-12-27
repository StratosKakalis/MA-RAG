import string
import random
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset
from src.agentic_rag_wrapper import agentic_rag_answer

# ---------------------------
# Utility functions
# ---------------------------

def normalize_text(text):
    """Lowercase, remove punctuation, articles, extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match(prediction, gold):
    return int(normalize_text(prediction) == normalize_text(gold))

def accuracy(prediction, gold):
    return int(normalize_text(prediction) == normalize_text(gold))

# ---------------------------
# Evaluation loop
# ---------------------------

def evaluate_benchmark(benchmark_name):
    """
    Evaluate agentic_rag_answer on the specified benchmark.
    Supports: SciFact, SciQ, ARC, MLQA, SCIENTIFIC-QA, MedQA
    """
    print(f"\nEvaluating {benchmark_name}...")
    
    options = []

    # Load dataset
    if benchmark_name.lower() == "scifact":
        dataset = load_dataset("allenai/scifact", "claims", split="validation", trust_remote_code=True)
        is_multiple_choice = True
        options = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
        options = ["Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking"]
    elif benchmark_name.lower() == "sciq":
        dataset = load_dataset("allenai/sciq", split="validation")  # or test if available
        is_multiple_choice = True
    elif benchmark_name.lower() == "arc":
        dataset = load_dataset("arc", "ARC-Challenge", split="test")
        is_multiple_choice = True
    elif benchmark_name.lower() == "mlqa":
        dataset = load_dataset("mlqa", split="test")  # replace with CS version if needed
        is_multiple_choice = False
    elif benchmark_name.lower() == "scientific-qa":
        dataset = load_dataset("scientific_qa", split="test")
        is_multiple_choice = False
    elif benchmark_name.lower() == "medqa":
        dataset = load_dataset("medqa", split="test")
        is_multiple_choice = True
    else:
        raise ValueError(f"Benchmark {benchmark_name} not supported.")
    
    total = 0
    em_score = 0
    acc_score = 0
    outputs = []
    for item in tqdm(dataset):
        # print (f"Processing item {item}")

        # Some items in scifact lack evidence_label and thus cannot be evaluated, skip them
        if benchmark_name.lower() == "scifact":
            question = item.get("question", item.get("claim", "")).strip()
            if (item["evidence_label"] == "" or item["evidence_label"] is None):
                gold = "NOT_ENOUGH_INFO"
                continue        # NOTE: SKIP NEI ITEMS BECAUSE THE DATASET TREATS ITS CORPUS AS THE ONLY SOURCE OF TRUTH, MEANING IF SOMETHING IS TRUE BUT NOT IN THE CORPUS IT IS LABELED AS NEI
            else:
                gold = item["evidence_label"]
        elif benchmark_name.lower() == "sciq":
            gold = item["correct_answer"]
            question = item["question"]
            options = [item["distractor1"], item["distractor2"], item["distractor3"], item["correct_answer"]]
            # Shuffle options to avoid position bias
            random.shuffle(options)

        total += 1

        # Format multiple-choice questions
        if is_multiple_choice:
            question += f"\nChoices:\n{options}\nPlease select the correct answer."
        
        # Call the Agentic RAG wrapper but add a timeout of 5 minutes that skips the item if exceeded
        pred = agentic_rag_answer(question)
        
        # Determine gold answer
        if is_multiple_choice:
            print(f"Question {question}, gold:{gold}, pred:{pred}")
            acc_score += accuracy(pred, gold)
            em_score += exact_match(pred, gold)
        else:
            gold_answers = item.get("answers") or [item.get("answer", "")]
            gold = gold_answers[0] if isinstance(gold_answers, list) else gold_answers
            em_score += exact_match(pred, gold)

        outputs.append({
            "question": question,
            "prediction": pred,
            "gold": gold
        })
    
    print (f"Skipped {invalid} invalid items.")
    em = em_score / total
    acc = acc_score / total if is_multiple_choice else None
    
    print(f"{benchmark_name} results:")
    print(f"  EM: {em:.4f}")
    if acc is not None:
        print(f"  Accuracy: {acc:.4f}")
    
    return {"benchmark": benchmark_name, "EM": em, "Accuracy": acc}

# ---------------------------
# Main evaluation
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Specify a single benchmark to run (optional)")
    args = parser.parse_args()
    
    benchmarks = ["SciFact", "SciQ", "ScienceQA", "ARC", "MLQA", "SCIENTIFIC-QA", "MedQA"]
    
    if args.benchmark:
        if args.benchmark not in benchmarks:
            raise ValueError(f"Unknown benchmark {args.benchmark}")
        benchmarks = [args.benchmark]
    
    results = []
    for bm in benchmarks:
        outputs, res = evaluate_benchmark(bm)
        results.append(res)

        # Save outputs under benchmark_outputs/{benchmark_name}.jsonl
        save_path = f"benchmark_outputs/{bm}_outputs.jsonl"
        os.makedirs("benchmark_outputs", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for out in outputs:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Saved outputs to {save_path}")
    
    # Summary
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['benchmark']}: EM={r['EM']:.4f}, Accuracy={r['Accuracy']}")