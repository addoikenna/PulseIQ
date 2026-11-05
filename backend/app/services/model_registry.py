"""PulseIQ V2.2 metadata-only registry.

The service-role client bypasses RLS. All public entrypoints require the
authenticated user ID; the SQL RPC repeats the ownership check atomically.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from app.services.supabase_client import get_supabase_client


class ModelRegistryError(Exception):
    pass


class ModelNotFoundError(ModelRegistryError):
    pass


def _uuid(value: str) -> str:
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid identifier.") from exc


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-safe values; reject NaN/Inf."""
    def default(obj: Any) -> Any:
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
    return json.loads(json.dumps(value, default=default, allow_nan=False))


def get_model(user_id: str, model_id: str) -> dict[str, Any]:
    response = (get_supabase_client().table("predictive_models")
                .select("*").eq("id", _uuid(model_id))
                .eq("user_id", _uuid(user_id)).limit(1).execute())
    if not response.data:
        raise ModelNotFoundError("Model not found or access denied.")
    return response.data[0]


def list_models(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    response = (get_supabase_client().table("predictive_models")
                .select("*").eq("user_id", _uuid(user_id))
                .order("created_at", desc=True).limit(min(max(limit, 1), 100)).execute())
    return response.data or []


def list_versions(user_id: str, model_id: str) -> list[dict[str, Any]]:
    get_model(user_id, model_id)  # Check owner before service-role access.
    response = (get_supabase_client().table("predictive_model_versions")
                .select("*").eq("model_id", _uuid(model_id))
                .eq("user_id", _uuid(user_id))
                .order("version_number", desc=True).execute())
    return response.data or []


def save_training_result(
    *, user_id: str, analysis_id: str, result: dict[str, Any],
    model_id: str | None = None, model_name: str | None = None,
) -> dict[str, Any]:
    """Atomically create model/version or append a version using a SQL RPC.

    The SQL function locks existing model rows, guaranteeing sequential version
    numbers even for concurrent backend workers. No estimator binary is saved.
    """
    user_id = _uuid(user_id)
    analysis_id = _uuid(analysis_id)
    owned_model_id = _uuid(model_id) if model_id else None
    label = (model_name or f'{result["target_column"]} prediction').strip()
    if not owned_model_id and not 1 <= len(label) <= 150:
        raise ValueError("Model name must contain 1 to 150 characters.")

    version_data = {
        "algorithm": result["best_model"],
        "selected_features": result["selected_features"],
        "training_config": {
            "test_size": result["test_size"],
            "random_state": result["random_state"],
            "training_rows": result["training_rows"],
            "test_rows": result["test_rows"],
            "automatically_removed_features": result.get("automatically_removed_features", []),
            "selection_metric": result["selection_metric"],
            "selection_method": result["selection_method"],
        },
        "evaluation_metrics": result["best_model_metrics"],
        "baseline": result["baseline"],
        "baseline_comparison": result["baseline_comparison"],
        "reliability": result["model_reliability"],
        "model_comparison": result["models"],
        "feature_importance": result.get("feature_importance", []),
        "warnings": result.get("warnings", []),
    }
    try:
        response = get_supabase_client().rpc("pulseiq_save_model_version", {
            "p_user_id": user_id,
            "p_analysis_id": analysis_id,
            "p_model_id": owned_model_id,
            "p_model_name": label,
            "p_target_column": result["target_column"],
            "p_problem_type": result["problem_type"],
            "p_version_data": _json_safe(version_data),
        }).execute()
        if not response.data:
            raise ModelRegistryError("Database did not return a saved version.")
        saved = response.data[0] if isinstance(response.data, list) else response.data
        return {
            "model_id": saved["model_id"],
            "version_id": saved["version_id"],
            "version_number": saved["version_number"],
        }
    except ModelRegistryError:
        raise
    except Exception as exc:
        # Preserve the current route's separate training/persistence failure boundary.
        raise ModelRegistryError(
            "Training completed, but model history could not be saved."
        ) from exc
