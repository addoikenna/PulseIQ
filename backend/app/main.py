from fastapi import FastAPI, UploadFile, File
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

    # Read CSV file
    df = pd.read_csv(file.file)

    # Generate dataset summary
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    return {
        "filename": file.filename,
        "summary": summary
    }
