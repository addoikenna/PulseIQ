from fastapi import UploadFile, HTTPException
import pandas as pd


ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_uploaded_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a CSV, XLSX, or XLS file.",
        )


def validate_file_size(file: UploadFile) -> None:
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Please upload a file under {MAX_FILE_SIZE_MB}MB for now.",
        )


def read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    validate_uploaded_file(file)
    validate_file_size(file)

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