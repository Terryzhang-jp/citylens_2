"""
CityLens API Server
FastAPI backend for urban scene analysis
"""

import os
import json
import asyncio
import uuid
import time
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the core analysis function
from src.graph_v7 import run_perception_v7_streaming
from src.utils.exif import extract_exif, format_exif_for_display
from src.utils.logger import main_logger

# Store for pending image uploads (for SSE flow)
# Format: {id: {image_data, lat, lon, has_markup, created_at}}
pending_analyses: dict[str, dict] = {}

# TTL for pending analyses (5 minutes)
PENDING_ANALYSIS_TTL = 300

# Cleanup interval (1 minute)
CLEANUP_INTERVAL = 60


async def cleanup_expired_analyses():
    """Background task to clean up expired pending analyses."""
    while True:
        try:
            await asyncio.sleep(CLEANUP_INTERVAL)
            now = time.time()
            expired_ids = [
                id for id, ctx in pending_analyses.items()
                if now - ctx.get("created_at", 0) > PENDING_ANALYSIS_TTL
            ]
            for id in expired_ids:
                del pending_analyses[id]
                main_logger.info(f"清理过期分析任务: {id[:8]}...")
            if expired_ids:
                main_logger.info(f"清理完成，移除 {len(expired_ids)} 个过期任务，当前待处理: {len(pending_analyses)}")
        except asyncio.CancelledError:
            main_logger.info("清理任务已停止")
            break
        except Exception as e:
            main_logger.error(f"清理任务出错: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks."""
    # Startup: start background cleanup task
    cleanup_task = asyncio.create_task(cleanup_expired_analyses())
    main_logger.info("启动后台清理任务")
    yield
    # Shutdown: cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    main_logger.info("后台清理任务已关闭")

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="CityLens API",
    description="AI-powered urban scene analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "CityLens API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check"""
    api_key = os.getenv("GEMINI_API_KEY")
    return {
        "status": "healthy",
        "gemini_configured": bool(api_key),
    }


@app.post("/api/upload")
async def upload_image(
    image: UploadFile = File(...),
    has_markup: bool = Form(False),
):
    """
    Upload image and get an ID for streaming analysis.

    Returns:
        - id: Unique ID for the analysis session
        - exif: Extracted EXIF data (gps, datetime, camera)
        - has_markup: Whether the image contains user markup
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_data = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    # Extract EXIF data
    exif_raw = extract_exif(image_data)
    exif_display = format_exif_for_display(exif_raw)

    # Generate unique ID and store analysis context with timestamp
    image_id = str(uuid.uuid4())
    pending_analyses[image_id] = {
        "image_data": image_data,
        "lat": exif_display.get("lat"),
        "lon": exif_display.get("lon"),
        "has_markup": has_markup,
        "created_at": time.time(),
    }
    main_logger.info(f"图片上传成功: {image_id[:8]}..., 待处理队列: {len(pending_analyses)}")

    return {
        "id": image_id,
        "exif": exif_display,
        "has_markup": has_markup,
    }


@app.get("/api/analyze/{image_id}")
async def analyze_stream(
    image_id: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
):
    """
    Stream analysis progress via Server-Sent Events.

    Query params:
    - lat: Override latitude (if user provided manually)
    - lon: Override longitude (if user provided manually)

    Events:
    - progress: {layer, message, detail}
    - result: {success, data}
    - error: {message}
    """
    if image_id not in pending_analyses:
        main_logger.warning(f"分析请求失败 - 图片不存在或已过期: {image_id[:8]}...")
        raise HTTPException(status_code=404, detail="Image not found or expired")

    analysis_ctx = pending_analyses.pop(image_id)
    main_logger.info(f"开始分析: {image_id[:8]}..., 剩余待处理: {len(pending_analyses)}")
    image_data = analysis_ctx["image_data"]

    # Use override location if provided, otherwise use EXIF
    final_lat = lat if lat is not None else analysis_ctx.get("lat")
    final_lon = lon if lon is not None else analysis_ctx.get("lon")
    has_markup = analysis_ctx.get("has_markup", False)

    async def event_stream():
        try:
            async for event in run_perception_v7_streaming(
                image_data=image_data,
                latitude=final_lat,
                longitude=final_lon,
                has_markup=has_markup,
            ):
                event_type = event.get("type", "progress")
                yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            error_event = {"type": "error", "message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
