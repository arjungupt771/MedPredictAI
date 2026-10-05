"""
MedPredict AI - LLM Plain-Language Explainer (optional)
Pipeline: structured ML prediction -> trusted knowledge retrieval -> LLM
rephrasing. The LLM is given ONLY the prediction, SHAP factors, and the fixed
knowledge snippet — it is instructed not to introduce outside medical claims,
and the app works fine without it (falls back to a template) if no API key
is configured, so this never becomes a hard dependency.
"""
import os

from knowledge_base import retrieve, DISCLAIMER

SYSTEM_PROMPT = (
    "You are rephrasing an existing machine-learning health-screening result "
    "into plain, calm language for a general audience. You are NOT diagnosing "
    "anyone. Use ONLY the facts given to you (the predicted condition, its "
    "confidence, the listed symptom factors, and the reference note) — do not "
    "add new medical claims, causes, or treatment advice beyond the reference "
    "note. Keep it to 3-4 short sentences. Always end by recommending "
    "professional medical evaluation."
)


def _build_user_prompt(prediction: dict, factors: list[dict], knowledge: str) -> str:
    factor_lines = "\n".join(
        f"- {factor['symptom']} ({factor['direction']}, {factor['strength'].lower()} influence)"
        for factor in factors[:5]
    )
    return (
        f"Predicted condition: {prediction['disease']}\n"
        f"Calibrated confidence: {prediction['confidence_pct']}%\n"
        f"Contributing symptom factors:\n{factor_lines}\n\n"
        f"Reference note on this condition: {knowledge}\n\n"
        "Write the plain-language explanation now."
    )


def _fallback_explanation(prediction: dict, factors: list[dict], knowledge: str) -> str:
    """Used when no LLM API key is configured — a template, not a placeholder error."""
    top_factor = factors[0]["symptom"] if factors else "the reported symptoms"
    return (
        f"The model's top match was {prediction['disease']} "
        f"(calibrated confidence: {prediction['confidence_pct']}%), driven mainly by {top_factor}. "
        f"{knowledge} {DISCLAIMER}"
    )


def generate_plain_language_explanation(prediction: dict, factors: list[dict]) -> dict:
    knowledge = retrieve(prediction["disease"])
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        return {
            "explanation": _fallback_explanation(prediction, factors, knowledge),
            "source": "template_fallback",
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(prediction, factors, knowledge)}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return {
            "explanation": text.strip() + f"\n\n{DISCLAIMER}",
            "source": "llm",
        }
    except Exception:
        # Never let the optional feature break the core app.
        return {
            "explanation": _fallback_explanation(prediction, factors, knowledge),
            "source": "template_fallback_after_error",
        }