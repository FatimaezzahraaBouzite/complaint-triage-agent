"""REST API.  uvicorn app.api:app --reload   then POST /triage {"narrative": "..."}"""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from triage import config
from triage.agent import TriageAgent

app = FastAPI(title="Complaint triage agent")
agent = TriageAgent(trace_path=config.TRACE_PATH)


class Complaint(BaseModel):
    narrative: str = Field(min_length=50)
    complaint_id: str = "manual"
    company: str = ""
    state: str = ""
    tags: str = ""
    submitted_via: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model": getattr(agent.llm, "model", "?")}


@app.post("/triage")
def triage(c: Complaint):
    r = agent.triage(c.model_dump())
    return {"filed": r.decision is not None, "decision": r.decision, "steps": r.steps, "latency_s": r.latency_s,
            "guardrail_rejections": r.rejections, "tools": [t["name"] for t in r.tool_calls]}
