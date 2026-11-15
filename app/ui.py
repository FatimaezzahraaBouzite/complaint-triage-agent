"""Demo UI.  streamlit run app/ui.py
Pick a real held-out complaint (or paste your own), watch the agent work, compare with what really happened."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from triage import config
from triage.agent import TriageAgent

st.set_page_config(page_title="Complaint triage agent", layout="wide")
st.title("Complaint triage agent")
st.caption("Real consumer complaints from the CFPB public database. The agent has never seen the held-out ones.")


@st.cache_resource
def get_agent():
    return TriageAgent(trace_path=config.TRACE_PATH)


agent = get_agent()
st.sidebar.caption(f"Model: {getattr(agent.llm, 'model', '?')}")

if st.sidebar.button("Load a random held-out complaint", type="primary"):
    con = sqlite3.connect(config.DB_PATH)
    st.session_state.case = pd.read_sql("SELECT * FROM complaints WHERE split='test' ORDER BY RANDOM() LIMIT 1", con).iloc[0].to_dict()
    con.close()
    st.session_state.pop("result", None)
case = st.session_state.get("case")

text = st.text_area("Complaint narrative", value=case["narrative"] if case else "", height=220,
                    placeholder="Load a real complaint from the sidebar, or paste one here.")
if st.button("Triage this complaint") and len(text.strip()) >= 50:
    c = dict(case) if case and text == case["narrative"] else {"complaint_id": "manual", "narrative": text.strip(), "tags": ""}
    with st.spinner("Agent is investigating..."):
        st.session_state.result, st.session_state.ran_on = agent.triage(c), c

res = st.session_state.get("result")
if res:
    c, d = st.session_state.ran_on, res.decision
    left, right = st.columns(2)
    with left:
        st.subheader("Agent decision")
        if d is None:
            st.error("The agent could not file this case. It goes to a human.")
        else:
            st.markdown(f"**Product:** {d['product']}  \n**Issue:** {d['issue']}  \n**Priority:** {d['priority']}  \n"
                        f"**Routed to:** {d['team']}  \n**Predicted outcome:** {d['predicted_outcome']}  \n"
                        f"**Risk flags:** {', '.join(d['risk_flags']) or 'none'}")
            st.markdown("**Why**")
            st.write(d["rationale"])
            st.markdown("**Draft reply to the consumer**")
            st.info(d["draft_reply"])
    with right:
        st.subheader("What actually happened")
        if c.get("product"):
            ok = lambda a, b: "✅" if d and a == b else "❌"
            st.markdown(f"**Product:** {c['product']} {ok(d and d['product'], c['product'])}  \n"
                        f"**Issue:** {c['issue']} {ok(d and d['issue'], c['issue'])}  \n"
                        f"**Company response:** {c['company_response']} {ok(d and d['predicted_outcome'], c['outcome'])}  \n"
                        f"**Company:** {c['company']}  \n**Received:** {c['date_received']}")
        else:
            st.write("Pasted text: no ground truth to compare with.")
        st.caption(f"{res.steps} LLM steps, {res.input_tokens + res.output_tokens:,} tokens, {res.latency_s}s, "
                   f"{res.rejections} guardrail rejection(s)")
    with st.expander(f"Tool trace ({len(res.tool_calls)} calls)"):
        st.json(res.tool_calls)
