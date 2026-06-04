from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from app.services.data_analyzer import analyze_dataframe
from app.services.data_cleaner import clean_dataframe
from app.services.response_formatter import format_analysis_response


app = FastAPI(
    title="PulseIQ API",
    description="Backend API for AI-powered dataset analysis and dashboard generation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Welcome to PulseIQ API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/sample-analysis")
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
                    "description": "Shows how sales changes over time.",
                },
                {
                    "chart_type": "bar",
                    "title": "Count by category",
                    "x_axis": "category",
                    "y_axis": "count",
                    "description": "Compares the number of records across category.",
                },
            ],
        },
        "report": {
            "insights": [
                "The dataset contains 6 rows and 5 columns.",
                "The data quality score is 98/100, which suggests the dataset is in good condition for analysis.",
                "There is 1 missing value in the dataset. This should be reviewed before analysis.",
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


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file.file, skipinitialspace=True)

    if filename.endswith(".xlsx"):
        return pd.read_excel(file.file, engine="openpyxl")

    if filename.endswith(".xls"):
        return pd.read_excel(file.file, engine="xlrd")

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload a CSV or Excel file.",
    )


@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    allowed_extensions = (".csv", ".xlsx", ".xls")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

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

        return format_analysis_response(summary, file.filename)

    except EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

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