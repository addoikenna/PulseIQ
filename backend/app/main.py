from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd

app = FastAPI(
    title="PulseIQ API",
    description="Backend API for AI-powered dataset analysis and dashboard generation.",
    version="0.1.0",
)

@app.get("/")
def root():
    return {"message": "Welcome to PulseIQ API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported for now.")

    try:
        df = pd.read_csv(file.file, skipinitialspace=True)

        # Treat empty spaces and common missing markers as missing values
        df = df.replace(r"^\s*$", pd.NA, regex=True)
        df = df.replace(["N/A", "NA", "null", "NULL", "None", "none", "?"], pd.NA)

        rows, columns = df.shape

        missing_values = df.isnull().sum().to_dict()
        total_missing = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()

        possible_date_columns = []
        for col in df.columns:
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                if converted.notna().sum() / max(len(df), 1) >= 0.7:
                    possible_date_columns.append(col)
            except Exception:
                pass

        summary_statistics = df.describe(include="all").fillna("").to_dict()

        data_quality_score = 100

        if total_missing > 0:
            data_quality_score -= min(30, int((total_missing / max(rows * columns, 1)) * 100))

        if duplicate_rows > 0:
            data_quality_score -= min(20, int((duplicate_rows / max(rows, 1)) * 100))

        data_quality_score = max(data_quality_score, 0)

        summary = {
            "rows": rows,
            "columns": columns,
            "column_names": list(df.columns),
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_values": missing_values,
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_rows,
            "numeric_columns": numeric_columns,
            "text_columns": text_columns,
            "possible_date_columns": possible_date_columns,
            "summary_statistics": summary_statistics,
            "data_quality_score": data_quality_score,
            "preview": df.head(5).fillna("").to_dict(orient="records"),
        }

        return {
            "filename": file.filename,
            "summary": summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")