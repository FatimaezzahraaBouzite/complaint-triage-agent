"""The README is generated from saved results. These tests render it from an empty project and from a full
(offline) pipeline run, so a template mistake fails here instead of on your machine."""
import json

from triage.llm import KNNPolicy

from .fixture import build_db


class FakeLLM(KNNPolicy):
    model = "fake-llm"  # recorded as an LLM run; behaves like the KNN policy


def test_readme_renders_with_nothing_run(tmp_path):
    from report import readme
    text = readme.render(tmp_path)
    assert "Results:** not run yet" in text and "python -m triage.data" in text
    assert "{{" not in text and "{%" not in text


def test_readme_renders_every_section_from_pipeline_outputs(tmp_path):
    from analysis import eda
    from evals import run_eval
    from report import readme
    from triage import baseline
    db = build_db(tmp_path / "data" / "complaints.db")
    eda.main(db, out=tmp_path / "analysis")
    baseline.main(db, out=tmp_path / "analysis" / "baseline_report.md")
    run_eval.main(per_product=3, llm=KNNPolicy(), db_path=db, out_dir=tmp_path / "evals")
    run_eval.main(per_product=3, llm=FakeLLM(), db_path=db, out_dir=tmp_path / "evals")

    path = readme.update(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "not run yet" not in text and "not available yet" not in text and "{{" not in text
    for expected in ("Results at a glance", "Files and filtering funnel", "Relief by product", "Run history",
                     "fake-llm", "evals/comparison.png", "What the LLM adds beyond the tools"):
        assert expected in text, expected
    assert (tmp_path / "evals" / "comparison.png").exists()
    # the numbers come from the saved files, not from the template
    run = json.loads(sorted((tmp_path / "evals" / "runs").glob("*fake-llm*.json"))[0].read_text())
    assert f"{run['summary']['agent']['product_accuracy']:.3f}" in text
    build = json.loads((tmp_path / "data" / "build_summary.json").read_text())
    assert f"{build['complaints']:,}" in text


def test_stale_results_are_flagged(tmp_path):
    from evals import run_eval
    from report import readme
    db = build_db(tmp_path / "data" / "complaints.db")
    run_eval.main(per_product=2, llm=FakeLLM(), db_path=db, out_dir=tmp_path / "evals")
    summary = tmp_path / "data" / "build_summary.json"
    s = json.loads(summary.read_text())
    s["fingerprint"] = "rebuilt"
    summary.write_text(json.dumps(s))
    assert "the data was rebuilt after this evaluation run" in readme.render(tmp_path)


def test_import_of_an_older_results_file(tmp_path):
    from evals import run_eval
    from report import readme
    db = build_db(tmp_path / "data" / "complaints.db")
    run_eval.main(per_product=2, llm=FakeLLM(), db_path=db, out_dir=tmp_path / "old")
    path = readme.import_run(tmp_path / "old" / "results.json", model="claude-haiku-5-5", root=tmp_path)
    record = json.loads(path.read_text())
    assert record["uses_llm"] and record["per_product"] == 2 and record["model"] == "claude-haiku-5-5"
    assert "claude-haiku-5-5" in readme.render(tmp_path)