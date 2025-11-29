# app.py
from fastapi import FastAPI
from api.router import router
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 1. Initialize FastAPI Application
app = FastAPI(
    title="XAI Visualizer Backend",
    description="Backend API implementing Singleton, Strategy, Observer, and Command patterns.",
    version="1.0.0"
)

# 2. Setup CORS Middleware for Frontend (CRITICAL for React integration)
# The React frontend will run on a different port (e.g., 3000), so we must allow cross-origin requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this to your React app's URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include the API router
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    # Start the server on http://127.0.0.1:8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)