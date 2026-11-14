"""Focused contract tests for the new staged upload/confirm flow.

Run from backend/: python -m pytest -q tests/test_preparation_pipeline.py
Uses fake Supabase operations; no live credentials are needed.
"""
import asyncio
from io import BytesIO
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

from app.routes import preparation_route as stage


class FakeTable:
    def __init__(self, db):
        self.db = db
        self.filters = {}
        self.mode = "select"
        self.payload = None

    def select(self, *args):
        self.mode = "select"
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def limit(self, *args):
        return self

    def insert(self, payload):
        self.mode, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.mode, self.payload = "update", payload
        return self

    def execute(self):
        if self.mode == "insert":
            self.db.analysis = dict(self.payload)
            return SimpleNamespace(data=[self.db.analysis])
        if self.mode == "update":
            if all(self.db.analysis.get(k) == v for k, v in self.filters.items()):
                self.db.analysis.update(self.payload)
                return SimpleNamespace(data=[self.db.analysis])
            return SimpleNamespace(data=[])
        rows = [self.db.analysis] if self.db.analysis else []
        rows = [r for r in rows if all(r.get(k) == v for k, v in self.filters.items())]
        return SimpleNamespace(data=rows)


class FakeBucket:
    def __init__(self, db):
        self.db = db

    def upload(self, path, file, file_options=None):
        self.db.objects[path] = file

    def remove(self, paths):
        for p in paths:
            self.db.objects.pop(p, None)

    def download(self, path):
        return self.db.objects[path]


class FakeDB:
    def __init__(self):
        self.analysis = None
        self.objects = {}
        self.storage = SimpleNamespace(from_=lambda bucket: FakeBucket(self))

    def table(self, name):
        assert name == "analyses"
        return FakeTable(self)


def test_upload_does_not_generate_dashboard_until_confirmation(monkeypatch):
    db = FakeDB()
    data = pd.DataFrame({"category": ["online", "ONLINE"], "value": [1, 2]})
    analyzer_calls = []
    monkeypatch.setattr(stage, "get_supabase_client", lambda: db)
    monkeypatch.setattr(stage, "read_uploaded_file", lambda uploaded: data.copy())
    monkeypatch.setattr(stage, "dataframe_to_parquet_bytes", lambda frame: b"PARQUET")
    monkeypatch.setattr(stage, "download_stored_dataset", lambda path: data.copy())
    monkeypatch.setattr(stage, "analyze_dataframe", lambda frame: analyzer_calls.append(True) or {"rows": len(frame)})
    monkeypatch.setattr(stage, "format_analysis_response", lambda summary, filename: {
        "status": "success", "overview": {"data_quality_score": 95, "rows": summary["rows"]},
        "filename": filename, "dashboard": {"kpis": []}, "report": {"insights": []},
    })

    upload = UploadFile(filename="Example.csv", file=BytesIO(b"category,value\nonline,1\nONLINE,2\n"))
    prepared = asyncio.run(stage.prepare_dataset(upload, user_id="user-a"))
    assert prepared["status"] == "prepared"
    assert not analyzer_calls
    assert db.analysis["prediction_ready"] is False
    assert db.analysis["analysis_json"]["status"] == "prepared"
    assert len(db.objects) == 2

    with pytest.raises(HTTPException) as denied:
        stage.confirm_dataset(stage.ConfirmAnalysisRequest(analysis_id=prepared["analysis_id"]),
                              user_id="user-b")
    assert denied.value.status_code == 404
    assert not analyzer_calls

    result = stage.confirm_dataset(stage.ConfirmAnalysisRequest(analysis_id=prepared["analysis_id"]),
                                   user_id="user-a")
    assert result["status"] == "success"
    assert len(analyzer_calls) == 1
    assert db.analysis["prediction_ready"] is True
    assert db.analysis["analysis_json"]["dashboard"] == {"kpis": []}

    again = stage.confirm_dataset(stage.ConfirmAnalysisRequest(analysis_id=prepared["analysis_id"]),
                                  user_id="user-a")
    assert again == result
    assert len(analyzer_calls) == 1
    assert any("/original/" in key for key in db.objects)
