"""LLM backends with one interface: complete(system, messages, tools) -> LLMResponse.

AnthropicLLM  the real agent (needs ANTHROPIC_API_KEY).
KNNPolicy     no LLM at all: a fixed nearest-neighbour policy that drives the same tools
              (search similar cases -> majority vote -> base rates -> submit). It makes the whole
              pipeline runnable and testable for free, and it is the baseline the LLM agent has to beat.
"""
import json
import os
import uuid
from collections import Counter
from dataclasses import dataclass, field

from . import config


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list = field(default_factory=list)  # [{"id", "name", "input"}]
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicLLM:
    def __init__(self, model=config.MODEL):
        import anthropic
        self.client, self.model = anthropic.Anthropic(), model

    def complete(self, system, messages, tools):
        r = self.client.messages.create(model=self.model, max_tokens=1500, system=system, tools=tools, messages=messages)
        return LLMResponse(
            text="".join(b.text for b in r.content if b.type == "text"),
            tool_calls=[{"id": b.id, "name": b.name, "input": b.input} for b in r.content if b.type == "tool_use"],
            input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens)


class KNNPolicy:
    model = "knn-policy (no LLM)"

    def complete(self, system, messages, tools):
        narrative = messages[0]["content"].split("Narrative:\n", 1)[-1]
        names, done = {}, {}
        for m in messages[1:]:
            for b in m["content"] if isinstance(m["content"], list) else []:
                if b["type"] == "tool_use":
                    names[b["id"]] = b["name"]
                elif b["type"] == "tool_result":
                    done[names[b["tool_use_id"]]] = json.loads(b["content"])  # keeps the latest result per tool
        call = lambda name, **inp: LLMResponse(tool_calls=[{"id": "knn_" + uuid.uuid4().hex[:8], "name": name, "input": inp}])

        if "search_similar_complaints" not in done:
            return call("search_similar_complaints", query=narrative)
        hits = done["search_similar_complaints"]["results"]
        if not hits:
            return LLMResponse(text="No similar past complaints found; this case needs a human.")
        votes = Counter()
        for h in hits:
            votes[h["product"]] += h["similarity"]
        product = votes.most_common(1)[0][0]
        issue = Counter(h["issue"] for h in hits if h["product"] == product).most_common(1)[0][0]
        if "get_outcome_stats" not in done:
            return call("get_outcome_stats", product=product, issue=issue)
        rates = done["get_outcome_stats"].get("outcome_rates") or {"explanation": 1.0}
        outcome = max(rates, key=rates.get)
        priority = "low"
        for p in done.get("submit_triage", {}).get("problems", []):  # rejected: raise to the required floor
            priority = p.get("minimum_priority", priority)
        return call("submit_triage", product=product, issue=issue, priority=priority, predicted_outcome=outcome,
                    rationale=f"Nearest past complaints are mostly '{product}' / '{issue}'; historically "
                              f"{rates[outcome]:.0%} of such cases closed with outcome '{outcome}'.",
                    draft_reply="Thank you for contacting us. We have received your complaint and assigned it to a "
                                "specialist team for review. We will investigate the facts you described and respond "
                                "in writing within 15 days.")


def get_llm():
    if os.getenv("ANTHROPIC_API_KEY") and os.getenv("AGENT_LLM", "anthropic") != "offline":
        return AnthropicLLM()
    return KNNPolicy()
