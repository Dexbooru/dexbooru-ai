from pydantic import BaseModel


class HealthResponse(BaseModel):
    qdrant: bool
    gemini: bool
    amqp: bool
