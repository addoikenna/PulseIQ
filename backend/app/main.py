from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pandas.errors import EmptyDataError, ParserError

from app.services.data_analyzer import analyze_dataframe
from app.services.data_cleaner import clean_dataframe
from app.services.response_formatter import format_analysis_response
from app.services.file_handler import read_uploaded_file
from app.schemas.analysis import AnalysisResponse
from app.routes.analyses import router as analyses_router


app = FastAPI(
    title="PulseIQ API",
    description="""
    PulseIQ is an AI-powered analytics platform that transforms CSV and Excel files
    into dashboards, KPIs, visualizations, and executive business reports.

    Features:
    - Dataset profiling
    - KPI generation
    - Smart dashboard recommendations
    - Interactive chart generation
    - Executive AI reporting
    - Saved analysis management
    """,
    version="0.2.0",
    contact={
        "name": "PulseIQ Support",
        "email": "support@pulseiq.ai",
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(analyses_router)


# ============================================================
# SYSTEM
# ============================================================

@app.get(
    "/",
    tags=["System"],
    summary="API Root",
    description="Returns a welcome message and confirms that the PulseIQ API is running.",
)
def root():
    return {
        "message": "Welcome to PulseIQ API",
        "version": "0.2.0",
    }


# ============================================================
# MONITORING
# ============================================================

@app.get(
    "/health",
    tags=["Monitoring"],
    summary="Health Check",
    description="Checks whether the API is healthy and available.",
)
def health_check():
    return {"status": "healthy"}


# ============================================================
# ANALYSIS
# ============================================================

@app.get(
    "/sample-analysis",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Sample Analysis Response",
    description="Returns a sample PulseIQ analysis response for frontend development and testing.",
)
def sample_analysis():
    return {
        "status": "success",
        "message": "This is a sample PulseIQ analysis response.",
        "filename": "sample_sales.csv",
        "overview": {
            "rows": 6,
            "columns": 5,
            "data_quality_score": 98,
            "total_missing_values": 1,
            "duplicate_rows": 0,
        },
        "dashboard": {
            "kpis": [],
            "charts": [],
            "filters": [],
            "chart_recommendations": [
                {
                    "chart_type": "line",
                    "title": "Sales Trend Over Time",
                    "x_axis": "date",
                    "y_axis": "sales",
                    "description": "Shows how sales change over time.",
                },
                {
                    "chart_type": "bar",
                    "title": "Count by Category",
                    "x_axis": "category",
                    "y_axis": "count",
                    "description": "Compares the number of records across categories.",
                },
            ],
        },
        "report": {
            "insights": [
                "The dataset contains 6 rows and 5 columns.",
                "The data quality score is 98/100.",
                "There is 1 missing value in the dataset.",
                "No duplicate rows were detected.",
            ],
            "data_quality": {
                "missing_values": {
                    "date": 0,
                    "product": 0,
                    "category": 0,
                    "sales": 0,
                    "quantity": 1,
                },
                "cleaning_report": {
                    "columns_renamed": {},
                    "missing_markers_converted": 1,
                    "text_values_trimmed": 2,
                    "numeric_columns_converted": [],
                    "date_columns_detected": ["date"],
                    "duplicates_detected": 0,
                    "warnings": [
                        "Column 'quantity' has 1 missing value(s) and may need review."
                    ],
                },
            },
            "recommendations": [
                "Review missing values before making business decisions.",
                "Use dashboard filters to explore trends and segments.",
                "Monitor top-performing categories and metrics regularly.",
            ],
        },
        "preview": [
            {
                "date": "2024-01-01",
                "product": "Laptop",
                "category": "Electronics",
                "sales": 500000,
                "quantity": 2,
            }
        ],
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["Analysis"],
    summary="Analyze Dataset",
    description="""
    Upload a CSV, XLSX, or XLS dataset and receive:

    - KPI recommendations
    - Interactive charts
    - Dashboard filters
    - Data quality assessment
    - Executive business report
    - AI-generated insights
    """,
)
async def analyze_dataset(file: UploadFile = File(...)):
    allowed_extensions = (".csv", ".xlsx", ".xls")

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded."
        )

    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a CSV, XLSX, or XLS file.",
        )

    try:
        df = read_uploaded_file(file)

        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file contains no data.",
            )

        cleaned_df, cleaning_report = clean_dataframe(df)

        summary = analyze_dataframe(cleaned_df)
        summary["cleaning_report"] = cleaning_report

        return format_analysis_response(
            summary,
            file.filename,
        )

    except EmptyDataError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    except ParserError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be parsed. Please check the file format.",
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while analyzing dataset: {str(e)}",
        )