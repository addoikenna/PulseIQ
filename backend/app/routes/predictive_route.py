from fastapi import APIRouter, Depends, HTTPException

from app.schemas.predictive import (
    PredictiveReadinessRequest,
    PredictiveReadinessResponse,
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