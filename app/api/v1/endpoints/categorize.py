from fastapi import APIRouter

from app.schemas import CategorizeRequest, CategorizeResponse
from utils.categorizer import categorize_food_text

router = APIRouter()


@router.post("/categorize", response_model=CategorizeResponse)
def categorize(payload: CategorizeRequest) -> CategorizeResponse:
    result = categorize_food_text(payload.text)
    # Always respond with the schema fields, even if RAG fails/falls back.
    category = str(result.get("category", "other"))
    try:
        probability = float(result.get("probability", 0.0))
    except Exception:
        probability = 0.0
    return CategorizeResponse(probability=probability, category=category)


