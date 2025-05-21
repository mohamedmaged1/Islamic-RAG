from fastapi import FastAPI,APIRouter
import os

base_router = APIRouter(
    prefix="/api/v1",
    tags=["api/v1"])

@base_router.get("/")

async def welcome():
        app_name = os.getenv("APP_NAME", "FastAPI Application")
        app_version = os.getenv("APP_VERSION", "1.0.0")
        return {
            "message": f"Welcome to {app_name}!",
            "version": app_version
        }
