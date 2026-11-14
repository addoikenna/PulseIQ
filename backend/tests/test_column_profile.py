import warnings
import pandas as pd
from app.services.column_profiler import profile_columns, looks_like_date


def test_mixed_dates_and_no_warnings():
    df = pd.DataFrame({
        'student_id': [f'GPA_{i:06d}' for i in range(120)],
        'enrollment_date': (['2017-09-01', '17-May-2014', '2018/09/01', '2015-01-12 '] * 30),
        'observation_date': (['2018/09/01', '2015-05-17', 'Unknown', '-'] * 30),
        'program': (['Online', 'Blended', 'On-campus', 'Hybrid'] * 30),
        'semester1_gpa': (['3.32', '4.05', '1.0', '2.1'] * 30),
        'final_cgpa': ([3.7, 3.63, 1.59, 2.4] * 30)
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = profile_columns(df)
    assert not caught, [str(w.message) for w in caught]
    assert set(result['date_columns']) == {'enrollment_date', 'observation_date'}
    assert 'student_id' in result['id_columns']
    assert 'final_cgpa' in result['numeric_columns']
    assert 'program' in result['categorical_columns']
    assert 'semester1_gpa' not in result['date_columns']
    assert list(df['enrollment_date'].head()) == ['2017-09-01','17-May-2014','2018/09/01','2015-01-12 ','2017-09-01']


def test_numeric_strings_not_dates():
    assert not looks_like_date(pd.Series(['45', '33', '2026', '12345']))
