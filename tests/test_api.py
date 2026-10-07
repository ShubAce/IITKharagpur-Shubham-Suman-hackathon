import time

import pytest
from fastapi.testclient import TestClient

from tremor.api.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app(mode="replay", speed=200_000.0)  # the whole 4-day replay in a few seconds
    with TestClient(app) as c:
        deadline = time.time() + 600
        while time.time() < deadline:
            st = c.get("/api/status").json()
            if st["replay"]["finished"] and not st["busy"]:
                break
            time.sleep(1)
        time.sleep(1.5)  # let the final batch's signals settle
        yield c


def test_status_reports_the_replay(client):
    st = client.get("/api/status").json()
    assert st["mode"] == "replay" and st["model"] == "tremor-encoder"
    assert st["counters"]["documents"] > 30_000


def test_events_are_ranked_by_impact_and_the_invasion_is_on_top(client):
    events = client.get("/api/events?limit=10").json()
    impacts = [e["impact_score"] for e in events]
    assert impacts == sorted(impacts, reverse=True)
    assert events[0]["event_type"] == "GEOPOLITICAL" and events[0]["impact_score"] >= 9
    detail = client.get(f"/api/events/{events[0]['event_id']}").json()
    assert detail["signal"]["impact_factors"] and detail["history"]


def test_module_b_stress_tests_were_triggered(client):
    runs = client.get("/api/stress/runs").json()
    assert runs, "a geopolitical event above 7 must trigger a stress test"
    assert any(r["trigger"]["event_type"] == "GEOPOLITICAL" for r in runs)
    full = client.get(f"/api/stress/runs/{runs[-1]['run_id']}").json()
    assert full["totals"]["value_after"] != full["totals"]["value_before"]


def test_module_a_index_rebalanced(client):
    snap = client.get("/api/index").json()
    assert snap["rebalances"] > 1
    assert sum(c["weight"] for c in snap["constituents"]) == pytest.approx(1.0, abs=1e-3)  # weights are rounded to 5 dp


def test_custom_stress_and_analyze(client):
    result = client.post("/api/stress/run", json={"shocks": {"EQ_US": -10, "IR_USD_10Y": 200}}).json()
    assert result["totals"]["pnl"] < 0
    out = client.post("/api/analyze", json={"text": "Moody's cuts Boeing to junk"}).json()
    assert {"sentiment_score", "event_type", "impact_score"} <= out.keys()
    assert client.post("/api/stress/run", json={"shocks": {"NOT_A_FACTOR": 1}}).status_code == 422


def test_credit_watchlist_flags_russian_obligors(client):
    watch = client.get("/api/watchlist").json()
    flagged = {e["entity_id"]: e for e in watch["entries"]}
    assert {"SBER", "GAZP"} & flagged.keys(), "Russian obligors must be under review during the invasion"
    first = watch["first_flagged"]["GAZP"]
    assert min(first.values()) < "2022-02-25"  # before S&P cut Russia to junk (25 Feb 2022)
    assert all(e["factors"] and e["status"] in ("Watch Negative", "Monitor") for e in watch["entries"])


def test_risk_memo_and_scenario_evidence(client):
    run = client.get("/api/stress/runs").json()[0]
    memo = client.get(f"/api/stress/runs/{run['run_id']}/memo?summary=template")  # no language model in tests
    assert memo.status_code == 200 and "Suggested actions" in memo.text and "CET1" in memo.text
    assert "<script" not in memo.text.lower()  # headlines are escaped, never executed
    assert client.get("/api/stress/runs/run_unknown/memo").status_code == 404
    evaluation = client.get("/api/evaluation").json()
    assert evaluation["scenario_backtest"]["summary"]["tremor"]["direction_hit_rate"] > 0.8


def test_replay_packs_include_both_crises(client):
    names = {p["name"] for p in client.get("/api/replay/packs").json()}
    assert {"ukraine_2022", "svb_2023"} <= names
