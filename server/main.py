from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from transformers import pipeline
import threading
import re

# ======== Модели ========
class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва для анализа")

class LengthOut(BaseModel):
    length: int

class SarcasmOut(BaseModel):
    label: str
    score: float
    is_sarcastic: bool
    model: str = "helinivan/english-sarcasm-detector"
    note: Optional[str] = None

class StatsOut(BaseModel):
    length_chars: int
    length_no_spaces: int
    word_count: int
    unique_words: int
    avg_word_length: float
    sentence_count: int

class FullAnalysisOut(BaseModel):
    text: str
    stats: StatsOut
    sarcasm: SarcasmOut

# ======== Приложение ========
app = FastAPI(
    title="Review Analyzer API",
    description="API для анализа отзывов (статистика + детектор сарказма)",
    version="0.3.0",
)

# ======== Инициализация и загрузка моделей ========
_model_lock = threading.Lock()
_sarcasm_classifier = None
_MODEL_ID = "helinivan/english-sarcasm-detector"

def get_sarcasm_pipeline():
    global _sarcasm_classifier
    if _sarcasm_classifier is None:
        with _model_lock:
            if _sarcasm_classifier is None:
                _sarcasm_classifier = pipeline(
                    task="text-classification",
                    model=_MODEL_ID,
                    tokenizer=_MODEL_ID,
                )
    return _sarcasm_classifier

# ======== Утилиты ========
def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text, flags=re.UNICODE)

def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

def compute_stats(text: str) -> StatsOut:
    words = _tokenize_words(text)
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    length_chars = len(text)
    length_no_spaces = len(text.replace(" ", ""))
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    sentence_count = len(_sentence_split(text)) if text.strip() else 0
    return StatsOut(
        length_chars=length_chars,
        length_no_spaces=length_no_spaces,
        word_count=word_count,
        unique_words=unique_words,
        avg_word_length=round(avg_word_length, 3),
        sentence_count=sentence_count,
    )

# ======== Маршруты ========
@app.post("/analyze/length", response_model=LengthOut, summary="Посчитать длину отзыва")
def analyze_length(payload: ReviewIn) -> LengthOut:
    """Возвращает количество символов во входном тексте (включая пробелы)."""
    return LengthOut(length=len(payload.text))

@app.post("/analyze/sarcasm", response_model=SarcasmOut, summary="Определить сарказм (EN)")
def analyze_sarcasm(payload: ReviewIn) -> SarcasmOut:
    clf = get_sarcasm_pipeline()
    result = clf(payload.text)[0]
    raw_label = result.get("label", "")
    score = float(result.get("score", 0.0))
    if raw_label in {"1", "LABEL_1", "SARCASTIC", "Sarcastic"}:
        label = "Sarcastic"
        is_sarcastic = True
    else:
        label = "Not Sarcastic"
        is_sarcastic = False
    note = "Модель для английского текста. Результаты могут быть неточными для других языков."
    return SarcasmOut(label=label, score=score, is_sarcastic=is_sarcastic, note=note)

@app.post("/analyze/full", response_model=FullAnalysisOut, summary="Полный анализ отзыва (статистика + сарказм)")
def analyze_full(payload: ReviewIn) -> FullAnalysisOut:
    """Выполняет полный анализ: базовая статистика + детекция сарказма (EN)."""
    stats = compute_stats(payload.text)
    sarcasm = analyze_sarcasm(payload)  # переиспользуем логику
    return FullAnalysisOut(text=payload.text, stats=stats, sarcasm=sarcasm)

@app.get("/")
def root():
    return {
        "service": "Review Analyzer API",
        "version": "0.3.0",
        "endpoints": {
            "POST /analyze/length": "Передайте JSON { 'text': '...' } и получите длину текста",
            "POST /analyze/sarcasm": "EN детектор сарказма на базе BERT (Hugging Face)",
            "POST /analyze/full": "Полный анализ: статистика + сарказм (EN)",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
