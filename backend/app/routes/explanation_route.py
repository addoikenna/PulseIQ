from fastapi import APIRouter, Depends, HTTPException

from app.schemas.explanation import (
    ChartExplanationResponse,
    ExplainChartRequest,
    ExplainKPIRequest,
    KPIExplanationResponse,
)
from app.services.auth import get_current_user_id
from app.services.kpi_chart_explainer import (
    explain_chart,
    explain_kpi,
)


router = APIRouter(
    prefix="/explain",
    tags=["KPI and Chart Explanations"],
)


@router.post(
    "/kpi",
    response_model=KPIExplanationResponse,
    summary="Explain a dashboard KPI",
)
def explain_dashboard_kpi(
    request: ExplainKPIRequest,
    user_id: str = Depends(get_current_user_id),
) -> KPIExplanationResponse:
    """
    Generate a grounded explanation for a KPI stored in a saved analysis.

    The KPI is retrieved from the saved analysis using its zero-based index.
    The authenticated user must own the analysis.
    """

    try:
        return explain_kpi(
            analysis_id=request.analysis_id,
            kpi_index=request.kpi_index,
            user_id=user_id,
        )

    except ValueError as error:
        error_message = str(error)

        if "index is outside" in error_message.lower():
            status_code = 400
        else:
            status_code = 404

        raise HTTPException(
            status_code=status_code,
            detail=error_message,
        )

    except Exception as error:
        print(f"Explain KPI endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate the KPI explanation.",
        )


@router.post(
    "/chart",
    response_model=ChartExplanationResponse,
    summary="Explain a dashboard chart",
)
def explain_dashboard_chart(
    request: ExplainChartRequest,
    user_id: str = Depends(get_current_user_id),
) -> ChartExplanationResponse:
    """
    Generate a grounded explanation for a chart stored in a saved analysis.

    The chart is retrieved from the saved analysis using its zero-based index.
    The authenticated user must own the analysis.
    """

    try:
        return explain_chart(
            analysis_id=request.analysis_id,
            chart_index=request.chart_index,
            user_id=user_id,
        )

    except ValueError as error:
        error_message = str(error)

        if "index is outside" in error_message.lower():
            status_code = 400
        else:
            status_code = 404

        raise HTTPException(
            status_code=status_code,
            detail=error_message,
        )

    except Exception as error:
        print(f"Explain chart endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to generate the chart explanation.",
        )