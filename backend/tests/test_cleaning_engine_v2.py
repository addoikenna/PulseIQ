import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from app.services.cleaning_engine import CleaningRule, apply_rules, profile


def test_map_text_numeric_and_no_mutation():
    frame = pd.DataFrame({'status':['Reliable','RELIABLE','reliable'],'churn':[0,1,0]})
    out, events = apply_rules(frame, [
        CleaningRule(operation='map_values',column='status',mapping={'RELIABLE':'Reliable','reliable':'Reliable'}),
        CleaningRule(operation='map_values',column='churn',mapping={'0':'No Churn','1':'Churn'}),
    ])
    assert out.status.tolist() == ['Reliable']*3
    assert out.churn.tolist() == ['No Churn','Churn','No Churn']
    assert frame.churn.tolist() == [0,1,0]
    assert [e['affected_rows'] for e in events] == [2,3]


def test_missing_and_conversion():
    frame = pd.DataFrame({'a':[1.0, None, 3.0], 'date':['2026-01-01','2026-01-02',None], 'flag':['Y','no',None]})
    out, events = apply_rules(frame, [
        CleaningRule(operation='impute_missing',column='a',strategy='median'),
        CleaningRule(operation='parse_date',column='date'),
        CleaningRule(operation='convert_type',column='flag',target_type='boolean'),
    ])
    assert out.a.tolist() == [1,2,3]
    assert str(out.date.dtype).startswith('datetime64')
    assert out.flag.tolist()[0] == True
    assert events[0]['affected_rows'] == 1


def test_rename_drop_and_duplicate():
    frame = pd.DataFrame({'old':['x','x','y'],'score':[1,1,9]})
    out, events = apply_rules(frame, [
        CleaningRule(operation='rename_column',column='old',new_column='category'),
        CleaningRule(operation='drop_duplicates',duplicate_columns=['category'],duplicate_keep='last'),
        CleaningRule(operation='drop_column',column='score'),
    ])
    assert list(out.columns)==['category'] and len(out)==2
    assert events[1]['affected_rows']==1


def test_strict_conversion_and_bad_rules():
    frame = pd.DataFrame({'mixed':['12','bad'],'score':[1.5,2.0]})
    with pytest.raises(ValueError, match='introduce'):
        apply_rules(frame,[CleaningRule(operation='convert_type',column='mixed',target_type='numeric')])
    with pytest.raises(ValueError, match='fractional'):
        apply_rules(frame,[CleaningRule(operation='convert_type',column='score',target_type='integer')])


def test_outlier_flag_and_clip():
    frame = pd.DataFrame({'x':[1,2,2,3,100]})
    out, events = apply_rules(frame,[CleaningRule(operation='flag_outliers',column='x',new_column='x_outlier')])
    assert bool(out.x_outlier.iloc[-1])
    assert events[0]['affected_rows'] == 1
    clipped, _ = apply_rules(frame,[CleaningRule(operation='clip_range',column='x',upper_bound=10)])
    assert clipped.x.iloc[-1]==10 and frame.x.iloc[-1]==100


def test_preview_profile():
    frame = pd.DataFrame({'x':['x',None]})
    assert profile(frame)['columns'][0]['missing_count']==1


def test_legacy_replace_missing():
    frame = pd.DataFrame({'x':[None,3]})
    changed, events=apply_rules(frame,[CleaningRule(operation='replace_missing',column='x',missing_value=0)])
    assert changed.x.tolist()==[0,3] and events[0]['affected_rows']==1
