from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype,
)

from app.schemas.predictive import (
    PredictiveReadinessRequest,
    PredictiveReadinessResponse,
    TrainPredictionRequest,
    PredictionTrainingResponse,
    SavedModelsResponse,
    SavedModelVersionsResponse,
    PredictiveColumnsResponse,
)
from app.services.auth import get_current_user_id
from app.services.data_storage import (
    DatasetNotAvailableError,
    DatasetStorageError,
    get_analysis_dataset,
)
from app.services.predictive_readiness import (
    assess_predictive_readiness,
)
from app.services.model_registry import (
    ModelRegistryError,
    ModelNotFoundError,
    save_training_result,
    get_model,
    list_models,
    list_versions,
)
from app.services.predictive_modeler import (
    PredictiveModelingError,
    train_predictive_models,
)


router = APIRouter(
    prefix="/predict",
    tags=["Predictive Analytics"],
)


@router.post(
    "/readiness",
    response_model=PredictiveReadinessResponse,
    summary="Assess predictive readiness for a saved analysis",
)
def assess_saved_analysis_readiness(
    request: PredictiveReadinessRequest,
    user_id: str = Depends(get_current_user_id),
) -> PredictiveReadinessResponse:
    """
    Assess whether the stored dataset behind a saved analysis
    is suitable for classification or regression.
    """

    try:
        analysis_record, dataframe = get_analysis_dataset(
            analysis_id=request.analysis_id,
            user_id=user_id,
        )

        result = assess_predictive_readiness(
            df=dataframe,
            analysis_id=request.analysis_id,
            filename=analysis_record.get("filename"),
        )

        return result

    except DatasetNotAvailableError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except DatasetStorageError as error:
        print(f"Predictive dataset storage error: {error}")

        raise HTTPException(
            status_code=503,
            detail=(
                "The stored dataset could not be retrieved. "
                "Please try again shortly."
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        print(f"Predictive readiness endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while assessing "
                "predictive readiness."
            ),
        )


@router.get(
    "/analyses/{analysis_id}/columns",
    response_model=PredictiveColumnsResponse,
    summary="List selectable columns for predictive modelling",
)
def get_predictive_columns(
    analysis_id: str,
    target_column: str | None = Query(default=None),
    user_id: str = Depends(get_current_user_id),
):
    """Return actual dataset column names after verifying analysis ownership.

    The readiness service supplies the existing exclusion recommendations.
    The frontend must never treat an eligibility hint as a bypass of training checks.
    """
    try:
        record, df = get_analysis_dataset(analysis_id=analysis_id, user_id=user_id)
        if target_column is not None and target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Selected target column does not exist.")

        readiness = assess_predictive_readiness(
            df=df, analysis_id=analysis_id, filename=record.get("filename")
        )
        # Readiness currently returns a dict; tolerate a Pydantic object too.
        if hasattr(readiness, "model_dump"):
            readiness = readiness.model_dump()
        excluded = {
            item["column"]: item["reason"]
            for item in readiness.get("excluded_columns", [])
        }
        candidate_targets = {
            item["column"]: item
            for item in readiness.get("candidate_targets", [])
        }
        columns = []
        for col in df.columns:
            series = df[col]
            if is_bool_dtype(series):
                dtype = "boolean"
            elif is_datetime64_any_dtype(series):
                dtype = "datetime"
            elif is_numeric_dtype(series):
                dtype = "numeric"
            elif pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(series):
                dtype = "categorical"
            else:
                dtype = "other"

            reason = excluded.get(col)
            if col == target_column:
                reason = "Selected prediction target; cannot also be a feature."
            # Recommended exclusions remain visible, not silently omitted.
            columns.append({
                "name": col,
                "data_type": dtype,
                "eligible_as_feature": reason is None,
                "eligible_as_target": col in candidate_targets and candidate_targets[col].get("problem_type") != "unsupported",
                "exclusion_reason": reason,
            })
        return {
            "analysis_id": analysis_id,
            "filename": record.get("filename"),
            "target_column": target_column,
            "columns": columns,
        }
    except HTTPException:
        raise
    except DatasetNotAvailableError as error:
        raise HTTPException(status_code=409, detail=str(error))
    except DatasetStorageError:
        raise HTTPException(status_code=503, detail="Could not retrieve the stored dataset.")
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        print(f"Predictive column metadata error: {error}")
        raise HTTPException(status_code=500, detail="Could not retrieve dataset columns.")


@router.post(
    "/train",
    response_model=PredictionTrainingResponse,
    summary="Train predictive models for a saved analysis",
)
def train_saved_analysis_model(
    request: TrainPredictionRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Train and compare classification or regression models using the
    stored dataset behind a saved analysis.
    """

    try:
        analysis_record, dataframe = get_analysis_dataset(
            analysis_id=request.analysis_id,
            user_id=user_id,
        )

        # Validate model identity BEFORE costly training. The atomic database
        # save function independently enforces this rule again at write time.
        if request.model_id is not None:
            existing = get_model(user_id=user_id, model_id=request.model_id)
            if existing["analysis_id"] != request.analysis_id:
                raise HTTPException(status_code=409, detail="Existing model belongs to a different dataset. Create a new model instead.")
            if existing["target_column"] != request.target_column:
                raise HTTPException(status_code=409, detail="Changing the prediction target requires creating a new model.")
            if request.problem_type is not None and existing["problem_type"] != request.problem_type:
                raise HTTPException(status_code=409, detail="Changing the prediction task requires creating a new model.")

        selected_features = None
        excluded_features = None

        if request.feature_selection:
            selected_features = (
                request.feature_selection.included_features
                or None
            )

            excluded_features = (
                request.feature_selection.excluded_features
                or None
            )

        result = train_predictive_models(
            dataframe=dataframe,
            target_column=request.target_column,
            problem_type=request.problem_type,
            selected_features=selected_features,
            excluded_features=excluded_features,
            test_size=request.test_size,
            random_state=request.random_state,
        )

        # Training is complete. Persistence is a separate, explicit failure boundary.
        # Never report saved=True unless Supabase confirms the version insert.
        saved_info = {}
        if request.save_model:
            saved_info = save_training_result(
                user_id=user_id,
                analysis_id=request.analysis_id,
                result=result,
                model_id=request.model_id,
                model_name=request.model_name,
            )

        return {
            "analysis_id": request.analysis_id,
            "filename": analysis_record.get("filename"),
            **result,
            **saved_info,
            "saved": bool(saved_info),
        }

    except HTTPException:
        raise

    except DatasetNotAvailableError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except DatasetStorageError as error:
        print(f"Predictive dataset storage error: {error}")

        raise HTTPException(
            status_code=503,
            detail=(
                "The stored dataset could not be retrieved. "
                "Please try again shortly."
            ),
        )

    except ModelNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))

    except ModelRegistryError as error:
        raise HTTPException(status_code=503, detail=str(error))

    except PredictiveModelingError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        print(f"Predictive training endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while training "
                "the predictive models."
            ),
        )

@router.get("/models", response_model=SavedModelsResponse, summary="List my saved predictive models")
def list_saved_predictive_models(user_id: str = Depends(get_current_user_id)):
    try:
        return {"models": list_models(user_id=user_id)}
    except Exception as error:
        print(f"Model registry listing error: {error}")
        raise HTTPException(status_code=503, detail="Could not retrieve saved models.")


@router.get("/models/{model_id}", summary="Get one of my saved predictive models")
def get_saved_predictive_model(model_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return get_model(user_id=user_id, model_id=model_id)
    except (ModelNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        print(f"Model registry retrieval error: {error}")
        raise HTTPException(status_code=503, detail="Could not retrieve the saved model.")


@router.get("/models/{model_id}/versions", response_model=SavedModelVersionsResponse, summary="Get saved model training history")
def get_saved_predictive_model_versions(model_id: str, user_id: str = Depends(get_current_user_id)):
    try:
        return {"model_id": model_id, "versions": list_versions(user_id=user_id, model_id=model_id)}
    except (ModelNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        print(f"Model version retrieval error: {error}")
        raise HTTPException(status_code=503, detail="Could not retrieve model versions.")
