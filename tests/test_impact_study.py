import pandas as pd

from tremor.training.impact_study import DAILY_PATH, analyse, company_sessions


def test_the_event_study_reproduces_from_committed_data():
    sessions = company_sessions(pd.read_csv(DAILY_PATH))
    result = analyse(sessions)
    assert result["with_news"] > 4000 and result["company_sessions"] >= result["with_news"]
    # Reported as found: the impact score is positively and significantly related to the size of the move ...
    impact = result["spearman"]["impact"]
    assert impact["rho"] > 0 and impact["p_value"] < 1e-3
    # ... and each session is compared with its market-adjusted move over the sessions around the news.
    assert sessions["abs_car"].between(0, 1).all()
    assert [b["bucket"] for b in result["buckets"]][0] == "no event"
