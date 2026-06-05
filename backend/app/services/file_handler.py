from fastapi import UploadFile, HTTPException
import pandas as pd


ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def validate_uploaded_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a CSV, XLSX, or XLS file.",
        )


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    validate_uploaded_file(file)

    filename = file.filename.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file.file, skipinitialspace=True)

    if filename.endswith(".xlsx"):
        return pd.read_excel(file.file, engine="openpyxl")

    if filename.endswith(".xls"):
        return pd.read_excel(file.file, engine="xlrd")

    raise HTTPException(
        status_code=400,
        detail="Unsupported file type. Please upload a CSV, XLSX, or XLS file.",
    )