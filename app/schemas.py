from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class CategorizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Menu item or food description")


class CategorizeResponse(BaseModel):
    probability: float = Field(..., description="Probability of the category")
    category: str = Field(..., description="Category of the food")


