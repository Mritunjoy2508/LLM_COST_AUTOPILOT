from __future__ import annotations
import random
from collections import Counter
from dataclasses import dataclass
from .models import MODEL_REGISTRY
from .providers import ProviderError, send_request

JUDGE_MODEL_KEY = "groq-mixtral-8x7b"
JUDGE_PROMPT_TEMPLATE = '''You are an impartial evaluator comparing two AI responses to the same prompt. Judge accuracy, relevance, and substance only. Do NOT prefer an answer just because it is longer; penalize unnecessary padding.\n\nOriginal prompt:\n{prompt}\n\nResponse A:\n{response_a}\n\nResponse B:\n{response_b}\n\nReply with ONLY: A, B, or TIE.'''

@dataclass
class JudgeVerdict:
    winner: str
    votes: list[str]
    position_bias_detected: bool

def _single_judge_call(prompt: str, response_a: str, response_b: str) -> str:
    try:
        result = send_request(MODEL_REGISTRY[JUDGE_MODEL_KEY], JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b), temperature=0.0, max_tokens=10)
        vote = result.text.strip().upper()
        return vote if vote in {"A", "B", "TIE"} else "TIE"
    except ProviderError:
        return "TIE"

def _majority(votes: list[str]) -> str:
    if not votes: return "TIE"
    counts = Counter(votes); m = max(counts.values())
    winners = [k for k,v in counts.items() if v == m]
    return winners[0] if len(winners) == 1 else "TIE"

def judge_pairwise(prompt: str, response_a: str, response_b: str, num_votes: int = 3) -> JudgeVerdict:
    votes = [_single_judge_call(prompt,response_a,response_b) for _ in range(num_votes)]
    swapped_raw = [_single_judge_call(prompt,response_b,response_a) for _ in range(num_votes)]
    flip = {"A":"B", "B":"A", "TIE":"TIE"}
    swapped_votes = [flip[v] for v in swapped_raw]
    orig, swapped = _majority(votes), _majority(swapped_votes)
    bias = orig != swapped and "TIE" not in (orig, swapped)
    winner = "TIE" if bias else _majority(votes + swapped_votes)
    return JudgeVerdict(winner, votes + swapped_votes, bias)

def spot_check_sample(judged_comparisons: list[dict], sample_size: int = 5) -> list[dict]:
    return random.sample(judged_comparisons, min(sample_size, len(judged_comparisons)))
