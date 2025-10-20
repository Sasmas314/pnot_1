from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

import threading
import re
import os

# ======== CatBoost ========
from catboost import CatBoostClassifier, Pool

# ======== Схемы ========
class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, description="Текст отзыва для анализа")

class LengthOut(BaseModel):
    length: int

class StatsOut(BaseModel):
    length_chars: int
    length_no_spaces: int
    word_count: int
    unique_words: int
    avg_word_length: float
    sentence_count: int

class SarcasmOut(BaseModel):
    label: str
    score: float
    is_sarcastic: bool
    model: str = "helinivan/english-sarcasm-detector"
    note: Optional[str] = None
    available: bool = True

class SentimentOut(BaseModel):
    label: str           # Positive | Neutral | Negative
    score: float         # top-confidence
    probs: Dict[str, float]
    model: str = "CatBoost review_pred.cbm"
    note: Optional[str] = "Offline CatBoost classifier for English text."
    threshold_neutral: float = 0.7

class FullAnalysisOut(BaseModel):
    text: str
    stats: StatsOut
    sentiment: SentimentOut
    sarcasm: SarcasmOut

# ======== Приложение ========
app = FastAPI(
    title="Review Analyzer API",
    description="API для анализа отзывов (CatBoost sentiment + статистика + сарказм*)\n*Сарказм включается, если доступен transformers/torch.",
    version="0.5.0",
)

# ======== Глобальные объекты и блокировки ========
_model_lock = threading.Lock()

# CatBoost
_catboost_model: Optional[CatBoostClassifier] = None
_CATBOOST_PATH = os.getenv("CATBOOST_MODEL_PATH", r"C:\Users\Asus\PycharmProjects\pnot_1\notebooks\catboost\review_pred.cbm")
_NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.7"))  # как в ноутбуке POSITIVE_THRESHOLD ~ 0.7

# Sarcasm (лениво и безопасно)
_sarcasm_pipeline = None
_sarcasm_init_tried = False
_SARCASM_MODEL_ID = "helinivan/english-sarcasm-detector"


# ======== Инициализация моделей ========
def get_catboost_model() -> CatBoostClassifier:
    """
    Ленивая загрузка CatBoost-модели для тональности.
    """
    global _catboost_model
    if _catboost_model is None:
        with _model_lock:
            if _catboost_model is None:
                if not os.path.exists(_CATBOOST_PATH):
                    raise FileNotFoundError(f"CatBoost model not found: {_CATBOOST_PATH}")
                model = CatBoostClassifier()
                model.load_model(_CATBOOST_PATH)
                _catboost_model = model
    return _catboost_model


def get_model_class_names(model: CatBoostClassifier) -> List[str]:
    """
    Извлечь имена классов из модели.
    Пытаемся сначала через public-атрибуты, затем через параметры.
    """
    # 1) привычный атрибут
    names = getattr(model, "classes_", None)
    if isinstance(names, (list, tuple)):
        return list(names)

    # 2) иногда лежит в metadata/параметрах
    try:
        params = model.get_all_params()
        if isinstance(params, dict):
            cn = params.get("class_names")
            if isinstance(cn, (list, tuple)):
                return list(cn)
    except Exception:
        pass

    # 3) fallback: предполагаем стандартный порядок
    return ["negative", "neutral", "positive"]


def get_sarcasm_pipeline_safe():
    """
    Пытается лениво создать transformers.pipeline для детекции сарказма.
    Возвращает (pipeline | None, note: str | None).
    Никогда не бросает исключения наружу.
    """
    global _sarcasm_pipeline, _sarcasm_init_tried
    if _sarcasm_pipeline is not None:
        return _sarcasm_pipeline, None

    if _sarcasm_init_tried:
        return None, "Sarcasm model is not available in this environment (transformers/torch missing)."

    with _model_lock:
        if _sarcasm_pipeline is not None:
            return _sarcasm_pipeline, None

        _sarcasm_init_tried = True
        try:
            from transformers import pipeline  # импорт только здесь
            pl = pipeline(
                task="text-classification",
                model=_SARCASM_MODEL_ID,
                tokenizer=_SARCASM_MODEL_ID,
            )
            _sarcasm_pipeline = pl
            return _sarcasm_pipeline, None
        except Exception as e:
            return None, f"Sarcasm model unavailable: {e.__class__.__name__}."


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

def preprocess_text(text: str) -> str:
    """
    Базовый препроцессинг (должен совпадать с тем, как кормили модель при обучении).
    Если в обучении были другие шаги — добавьте их здесь.
    """
    text = re.sub(r"\s+", " ", text.strip())
    return text.lower()


# ======== Эндпоинты ========
@app.post("/analyze/length", response_model=LengthOut, summary="Посчитать длину отзыва")
def analyze_length(payload: ReviewIn) -> LengthOut:
    """Возвращает количество символов (включая пробелы)."""
    return LengthOut(length=len(payload.text))


@app.post("/analyze/sentiment", response_model=SentimentOut, summary="Определить тональность (CatBoost)")
def analyze_sentiment(payload: ReviewIn) -> SentimentOut:
    """
    Классификация тональности через CatBoost-модель.
    Возвращает: Positive / Neutral / Negative + score (+ все вероятности).
    Введён порог нейтральности: если max(proba) < _NEUTRAL_THRESHOLD → Neutral.
    """
    model = get_catboost_model()
    text = preprocess_text(payload.text)

    # Используем Pool с text_features, как обычно в CatBoost текстовых задачах
    pool = Pool(
        data=[[text]],       # двумерная структура: один объект, один текстовый признак
        text_features=[0]    # признак 0 — текстовый
    )

    probas: List[List[float]] = model.predict_proba(pool)
    # predict_proba возвращает список для каждого объекта
    p = probas[0]
    class_names = get_model_class_names(model)

    # Складываем вероятности в словарь {class: proba}
    probs_dict: Dict[str, float] = {}
    for idx, cname in enumerate(class_names):
        probs_dict[str(cname).lower()] = float(p[idx])

    # Нормализуем ключи и приведём к нужным названиям
    rename = {
        "neg": "Negative", "negative": "Negative", "0": "Negative",
        "neu": "Neutral",  "neutral": "Neutral",   "1": "Neutral",
        "pos": "Positive", "positive": "Positive", "2": "Positive",
    }
    pretty_probs = {
        rename.get(k, k.capitalize()): v for k, v in probs_dict.items()
    }

    # Определяем итоговый класс
    top_label_raw = max(pretty_probs, key=pretty_probs.get)
    top_score = pretty_probs[top_label_raw]

    # Порог для нейтрали
    if top_score < _NEUTRAL_THRESHOLD:
        final_label = "Neutral"
        final_score = float(pretty_probs.get("Neutral", top_score))
    else:
        final_label = top_label_raw
        final_score = float(top_score)

    # Гарантируем наличие всех трёх ключей в probs
    for k in ("Negative", "Neutral", "Positive"):
        pretty_probs.setdefault(k, 0.0)

    return SentimentOut(
        label=final_label,
        score=final_score,
        probs=pretty_probs,
        threshold_neutral=_NEUTRAL_THRESHOLD,
    )


@app.post("/analyze/sarcasm", response_model=SarcasmOut, summary="Определить сарказм (EN, если доступно)")
def analyze_sarcasm(payload: ReviewIn) -> SarcasmOut:
    """
    Пытается использовать transformers-модель для сарказма.
    Если окружение не позволяет — возвращает доступное объяснение без падения сервиса.
    """
    pl, note_unavailable = get_sarcasm_pipeline_safe()
    if pl is None:
        return SarcasmOut(
            label="Not Available",
            score=0.0,
            is_sarcastic=False,
            note=note_unavailable or "Sarcasm model not available in this environment.",
            available=False,
        )

    result = pl(payload.text)[0]
    raw_label = str(result.get("label", "")).upper()
    score = float(result.get("score", 0.0))

    if raw_label in {"1", "LABEL_1", "SARCASTIC", "SARCASM", "SARCASTIC_1"}:
        label = "Sarcastic"
        is_sarcastic = True
    else:
        label = "Not Sarcastic"
        is_sarcastic = False

    note = "Модель для английского текста; результаты для других языков могут быть неточными."
    return SarcasmOut(label=label, score=score, is_sarcastic=is_sarcastic, note=note, available=True)


@app.post("/analyze/full", response_model=FullAnalysisOut, summary="Полный анализ (статистика + тональность + сарказм*)")
def analyze_full(payload: ReviewIn) -> FullAnalysisOut:
    """
    Выполняет полный анализ:
    - базовая статистика
    - тональность через CatBoost
    - сарказм (если доступна transformers/torch)
    """
    stats = compute_stats(payload.text)
    sentiment = analyze_sentiment(payload)
    sarcasm = analyze_sarcasm(payload)  # безопасно
    return FullAnalysisOut(text=payload.text, stats=stats, sentiment=sentiment, sarcasm=sarcasm)


@app.get("/", summary="Инфо")
def root():
    return {
        "service": "Review Analyzer API",
        "version": "0.5.0",
        "endpoints": {
            "POST /analyze/length": "JSON { 'text': '...' } → длина текста",
            "POST /analyze/sentiment": "CatBoost sentiment: Positive / Neutral / Negative (с вероятностями)",
            "POST /analyze/sarcasm": "EN сарказм (если доступна transformers/torch); не критично",
            "POST /analyze/full": "Статистика + тональность (CatBoost) + сарказм*",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
        "notes": [
            f"CatBoost model path: {_CATBOOST_PATH}",
            f"Neutral threshold: {_NEUTRAL_THRESHOLD}",
            "Sarcasm detector запускается только при наличии transformers/torch.",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
