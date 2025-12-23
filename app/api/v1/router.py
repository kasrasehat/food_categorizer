import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.endpoints import categorize, health, process_file

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Startup
    try:
        from utils.lfspan import get_vectordb_state

        logger.info("Pre-loading vector databases...")
        state = get_vectordb_state()
        # Make state available to request handlers via request.app.state
        app.state.vectordb_state = state
        logger.info("Successfully loaded vector databases: %s", list(state.keys()))
    except Exception as e:
        # Don't fail the startup if vector DB loading fails
        logger.warning("Failed to pre-load vector databases: %s", e, exc_info=True)

    yield


api_router = APIRouter()

# Route to redirect root to API documentation
@api_router.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


api_router.include_router(health.router, prefix="/api/v1", tags=["health"])
api_router.include_router(categorize.router, prefix="/api/v1", tags=["categorize"])
api_router.include_router(process_file.router, prefix="/api/v1", tags=["process-file"])


