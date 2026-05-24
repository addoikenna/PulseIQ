from fastapi import FastAPI

app = FastAPI(
    title="PulseIQ API",
    description="Backend API for AI-powered dataset analysis and dashboard generation.",
    version="0.1.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to PulseIQ API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
