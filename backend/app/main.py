from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd

from app.services.data_analyzer import analyze_dataframe

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
        summary = analyze_dataframe(df)

        return {
            "filename": file.filename,
            "summary": summary,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")