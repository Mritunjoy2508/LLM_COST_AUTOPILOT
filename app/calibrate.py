from __future__ import annotations
import json
from itertools import combinations
from pathlib import Path
from .eval_tasks import run_objective_eval
from .llm_judge import judge_pairwise, JUDGE_MODEL_KEY
from .models import MODEL_REGISTRY, TaskCategory
from .providers import send_request, ProviderError

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "calibrated_scores.json"
SUBJECTIVE_PROMPTS = {
    TaskCategory.CREATIVE_WRITING: "Write a short, vivid opening paragraph for a mystery novel.",
    TaskCategory.SUMMARIZATION: "Summarize in 2 sentences: The committee reviewed twelve proposals over three days, ultimately selecting four for further funding based on feasibility, budget alignment, and projected community impact, while the remaining eight were deferred pending revisions.",
    TaskCategory.GENERAL_QA: "What causes the seasons to change on Earth?",
    TaskCategory.ANALYSIS: "What are the main trade-offs between working remotely and working in an office?",
}

def run_objective_pass():
    scores = {}
    for key, model in MODEL_REGISTRY.items():
        print(f"Objective eval: {model.display_name}...")
        try:
            scores[key] = {cat.value: score for cat, score in run_objective_eval(model, send_request).items()}
        except Exception as exc:
            print(f"  skipped ({exc})"); scores[key] = {}
    return scores

def run_subjective_tournament():
    models = [(k,m) for k,m in MODEL_REGISTRY.items() if k != JUDGE_MODEL_KEY]
    wins = {k:{} for k,_ in models}; matches = {k:{} for k,_ in models}
    for category, prompt in SUBJECTIVE_PROMPTS.items():
        ck = category.value; print(f"Subjective tournament: {ck}...")
        responses = {}
        for key, model in models:
            try: responses[key] = send_request(model,prompt,temperature=0.7,max_tokens=200).text
            except ProviderError: responses[key] = None
        for (a,_),(b,_) in combinations(models,2):
            if responses[a] is None or responses[b] is None: continue
            verdict = judge_pairwise(prompt,responses[a],responses[b],num_votes=3)
            for k in (a,b): wins[k].setdefault(ck,0.0); matches[k].setdefault(ck,0); matches[k][ck]+=1
            if verdict.winner == "A": wins[a][ck]+=1
            elif verdict.winner == "B": wins[b][ck]+=1
            else: wins[a][ck]+=0.5; wins[b][ck]+=0.5
    return {k:{ck:round(10*w/matches[k][ck],2) for ck,w in cats.items() if matches[k][ck]>0} for k,cats in wins.items()}

def calibrate_all():
    print(f"NOTE: judge model is {JUDGE_MODEL_KEY}; excluded from the candidate tournament.\n")
    objective = run_objective_pass(); subjective = run_subjective_tournament()
    merged = {k:{**objective.get(k,{}), **subjective.get(k,{})} for k in MODEL_REGISTRY}
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(merged,indent=2),encoding="utf-8")
    print(f"\nCalibrated scores written to {OUTPUT_PATH}")
    return merged

if __name__ == "__main__": calibrate_all()
