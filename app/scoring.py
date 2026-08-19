from __future__ import annotations

from dataclasses import dataclass

from .models import ModelConfig, TaskCategory, all_models


DEFAULT_USABILITY_WEIGHT = 0.65
DEFAULT_COST_WEIGHT = 0.35


@dataclass
class ScoredModel:
    model_key: str
    model: ModelConfig
    capability_component: float
    cost_component: float
    score: float
    explanation: str


def _cost_component(model: ModelConfig) -> float:
    return min(10.0, model.verbosity_factor * 5.0)


def score_model(
    model: ModelConfig,
    task_category: TaskCategory,
    complexity: int,
    usability_weight: float = DEFAULT_USABILITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
) -> ScoredModel:
    raw_capability = model.capability_for(task_category)

    complexity_multiplier = 0.5 + (complexity / 10)
    capability_component = min(
        10.0,
        raw_capability * complexity_multiplier,
    )

    cost_component = _cost_component(model)

    score = (
        usability_weight * capability_component
        - cost_weight * cost_component
    )

    explanation = (
        f"capability={raw_capability:.1f}/10 "
        f"for {task_category.value} "
        f"(x{complexity_multiplier:.2f} "
        f"for complexity={complexity}) "
        f"-> {capability_component:.1f}; "
        f"cost_penalty={cost_component:.1f}/10 "
        f"(verbosity={model.verbosity_factor}x)"
    )

    return ScoredModel(
        model_key=model.model_id,
        model=model,
        capability_component=round(capability_component, 2),
        cost_component=round(cost_component, 2),
        score=round(score, 3),
        explanation=explanation,
    )


def rank_models(
    task_category: TaskCategory,
    complexity: int,
    usability_weight: float = DEFAULT_USABILITY_WEIGHT,
    cost_weight: float = DEFAULT_COST_WEIGHT,
) -> list[ScoredModel]:
    scored = [
        score_model(
            model,
            task_category,
            complexity,
            usability_weight,
            cost_weight,
        )
        for model in all_models()
    ]

    return sorted(
        scored,
        key=lambda scored_model: scored_model.score,
        reverse=True,
    )
