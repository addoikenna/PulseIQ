from fastapi import APIRouter, Depends, HTTPException

from app.schemas.predictive import (
    PredictiveReadinessRequest,
    PredictiveReadinessResponse,
    TrainPredictionRequest,
    PredictionTrainingResponse,
    SavedModelsResponse,
    SavedModelVersionsResponse,
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