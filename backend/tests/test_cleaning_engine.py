import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from app.services.cleaning_engine import CleaningRule, apply_rules, profile


def test_category_mapping_and_original_preserved():
    df = pd.DataFrame({"internet_access": ["Reliable", "RELIABLE", "reliable", "Poor"], "gpa": [3.1, None, 2.2, 3.4]})
    rules = [CleaningRule(operation="map_values", column="internet_access",
                          mapping={"RELIABLE": "Reliable", "reliable": "Reliable"})]
    changed, events = apply_rules(df, rules)
    assert changed.internet_access.tolist() == ["Reliable", "Reliable", "Reliable", "Poor"]
    assert df.internet_access.tolist() == ["Reliable", "RELIABLE", "reliable", "Poor"]
    assert events[0]["affected_rows"] == 2
    assert profile(changed)["columns"][0]["unique_count"] == 2


def test_duplicate_and_missing():
    df = pd.DataFrame({"cat": [" a ", "a", "a"], "num": [None, 2, 2]})
    changed, events = apply_rules(df, [
        CleaningRule(operation="trim", column="cat"),
        CleaningRule(operation="replace_missing", column="num", missing_value=2),
        CleaningRule(operation="drop_duplicates")])
    assert len(changed) == 1
    assert [x["affected_rows"] for x in events] == [1, 1, 2]
