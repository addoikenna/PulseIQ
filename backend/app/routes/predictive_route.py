from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from pandas.errors import EmptyDataError, ParserError

from app.schemas.predictive import PredictiveReadinessResponse
from app.services.auth import get_current_user_id
from app.services.data_cleaner import clean_dataframe
from app.services.file_handler import read_uploaded_file
from app.services.predictive_readiness import (
    assess_predictive_readiness,
)
from app.services.supabase_client import get_supabase_client


router = APIRouter(
    prefix="/predict",
    tags=["Predictive Analytics"],
)


ALLOWED_EXTENSIONS = (
    ".csv",
    ".xlsx",
    ".xls",
)


def verify_analysis_ownership(
    analysis_id: str,
    user_id: str,
) -> dict[str, Any]:
    """
    Confirm that the authenticated user owns the supplied saved analysis.
    """

    supabase = get_supabase_client()

    response = (
        supabase.table("analyses")
        .select("id, user_id, filename")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or access denied.",
        )

    return response.data[0]


@router.post(
    "/readiness",
    response_model=PredictiveReadinessResponse,
    summary="Assess dataset readiness for predictive modelling",
)
async def assess_dataset_readiness(
    file: UploadFile = File(
        ...,
        description="CSV or Excel dataset to assess.",
    ),
    analysis_id: str | None = Form(
        default=None,
        description=(
            "Optional ID of an existing saved analysis associated "
            "with the uploaded dataset."
        ),
    ),
    user_id: str = Depends(get_current_user_id),
) -> PredictiveReadinessResponse:
    """
    Assess whether an uploaded dataset is suitable for classification
    or regression.

    The endpoint identifies:

    - possible prediction targets;
    - likely prediction problem types;
    - excluded identifier and personal-information columns;
    - missing-data and duplication concerns;
    - class imbalance;
    - predictive-readiness score;
    - warnings and recommended next actions.

    When analysis_id is supplied, the authenticated user must own it.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. Please upload a CSV, "
                "XLSX, or XLS file."
            ),
        )

    if analysis_id:
        verify_analysis_ownership(
            analysis_id=analysis_id,
            user_id=user_id,
        )

    try:
        dataframe = read_uploaded_file(file)

        if dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file contains no data.",
            )

        cleaned_dataframe, cleaning_report = clean_dataframe(
            dataframe
        )

        if cleaned_dataframe.empty:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable data remained after the dataset "
                    "was cleaned."
                ),
            )

        result = assess_predictive_readiness(
            df=cleaned_dataframe,
            analysis_id=analysis_id,
            filename=file.filename,
        )

        cleaning_warnings = cleaning_report.get(
            "warnings",
            [],
        )

        if isinstance(cleaning_warnings, list):
            result["warnings"] = list(
                dict.fromkeys(
                    result.get("warnings", [])
                    + [
                        str(warning)
                        for warning in cleaning_warnings
                        if warning
                    ]
                )
            )

        return result

    except EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    except ParserError:
        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file could not be parsed. "
                "Please check the file format."
            ),
        )

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
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