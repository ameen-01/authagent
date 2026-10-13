"""Gradio demo: pick a patient and a configuration, watch the checklist, evidence and trace."""

from __future__ import annotations

import json

from .service import authorize, get_record, patient_ids

ICON = {"met": "MET", "unmet": "NOT MET", "missing": "MISSING"}


def _run(patient: str, config: str, llm_name: str):
    rec = get_record(patient)
    res = authorize(rec["chart"], config=config, llm_name=llm_name)
    gold = rec["gold"]["criteria"]
    table = []
    for c in res["policy"]["criteria"]:
        d = res["decisions"].get(c["id"], {"status": "missing", "evidence_ids": []})
        table.append([c["id"], c["text"], ICON[d["status"]], ", ".join(d.get("evidence_ids", [])),
                      ICON[gold[c["id"]]["status"]], gold[c["id"]]["variant"]])
    summary = (f"### Decision: **{res['decision'].upper()}** (gold: {rec['gold']['decision']})\n"
               f"Steps: {res['steps']} · LLM calls: {res['usage'].get('llm_calls', 0)} · "
               f"Verifier checks: {res['verifier_calls']} (sent to LLM: {res['escalations']}) · "
               f"Tokens: {res['usage'].get('prompt_tokens', 0) + res['usage'].get('completion_tokens', 0)}")
    lines = []
    for s in res["trace"]:
        if "tool" in s:
            thought = f" — _{s['thought']}_" if s.get("thought") else ""
            lines.append(f"**{s.get('step', '-')} · {s['tool']}**{thought}\n```\n{json.dumps(s.get('args', s.get('values', {})))}\n"
                         f"→ {json.dumps(s['observation'])[:600]}\n```")
        else:
            lines.append(f"**{s.get('step')} · parse error** {s.get('error')}")
    return summary, table, res["draft"], "\n".join(lines) or "_No tool trace for this configuration._"


def launch(share: bool = False, llm_name: str = "api") -> None:
    import gradio as gr

    with gr.Blocks(title="AuthAgent") as demo:
        gr.Markdown("# AuthAgent\nAgentic prior-authorization review on synthetic charts. Identifiers are redacted "
                    "before any model sees the chart.")
        with gr.Row():
            patient = gr.Dropdown(patient_ids(), label="Patient", value=patient_ids()[0])
            config = gr.Radio(["single_prompt", "fixed_pipeline", "agent", "agent_cascade"], value="agent_cascade",
                              label="Configuration")
            run = gr.Button("Run", variant="primary")
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["ID", "Criterion", "Predicted", "Evidence", "Gold", "Gold variant"],
                             wrap=True, interactive=False)
        with gr.Row():
            draft = gr.Textbox(label="Output document", lines=18)
            trace = gr.Markdown(label="Agent trace")
        run.click(lambda p, c: _run(p, c, llm_name), [patient, config], [summary, table, draft, trace])
    demo.launch(share=share)
