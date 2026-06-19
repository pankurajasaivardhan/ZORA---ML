import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "api"))
sys.path.insert(0, str(BASE_DIR / "modules" / "fraud"))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router


def get_logger_safe(name):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger_safe("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SENTINEL-ML API starting up")
    from dependencies import get_fraud_predictor, get_loan_predictor, get_health_predictor, get_equipment_predictor, get_network_predictor
    for name, loader in [
        ("Fraud", get_fraud_predictor),
        ("Loan", get_loan_predictor),
        ("Health", get_health_predictor),
        ("Equipment", get_equipment_predictor),
        ("Network", get_network_predictor)
    ]:
        try:
            loader()
            logger.info(f"{name} predictor loaded")
        except Exception as e:
            logger.error(f"{name} predictor failed: {e}")
    yield
    logger.info("SENTINEL-ML API shutting down")


app = FastAPI(
    title="SENTINEL-ML",
    description="Unified Multi-Domain Intelligent Risk Detection Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=str(BASE_DIR / "api")
    )