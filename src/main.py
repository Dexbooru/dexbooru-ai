import uvicorn
from fastapi import FastAPI

from utils.config import settings

app = FastAPI(title=settings.server_name, version="0.1.0")


def main() -> None:
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.server_port,
        reload=settings.environment == "development",
    )


if __name__ == "__main__":
    main()
