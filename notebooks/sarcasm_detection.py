import os
import math
import signal
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ========= ПАРАМЕТРЫ =========
INPUT_PATH  = r"C:\Users\Asus\PycharmProjects\pnot_1\data\raw\Reviews.csv"
OUTPUT_PATH = r"C:\Users\Asus\PycharmProjects\pnot_1\data\processed\Reviews_with_sarcasm.csv"

TEXT_COLUMN   = "Text"
BATCH_SIZE    = 32          # размер микро-батча для модели
PREFETCH      = 16          # сколько микро-батчей готовить (токенизировать) вперёд
MAX_LENGTH    = 256
THRESHOLD     = 0.5
NUM_WORKERS   = max(1, (os.cpu_count() or 4))  # все потоки
MODEL_NAME    = "helinivan/english-sarcasm-detector"
SAVE_EVERY    = 10_000      # сохранять каждые N обработанных строк
# =============================

# Глобальный флаг для мягкой остановки по Ctrl+C/сигналу
STOP_REQUESTED = False
def _set_stop_requested(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True
signal.signal(signal.SIGINT, _set_stop_requested)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _set_stop_requested)

def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data(input_path: str, text_col: str) -> pd.DataFrame:
    assert os.path.exists(input_path), f"File not found: {input_path}"
    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found. Available: {list(df.columns)}")
    # Индекс для резюма
    if "row_id" not in df.columns:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    return df

def resume_merge(base_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Если есть частичный результат — подмердживаем, чтобы не пересчитывать.
    """
    if not os.path.exists(output_path):
        return base_df.copy()
    out = pd.read_csv(output_path)
    if "row_id" not in out.columns:
        # если старый файл без row_id — игнорируем
        return base_df.copy()
    # merge по row_id, чтобы не потерять исходные колонки
    merged = base_df.merge(
        out[["row_id", "sarcasm_score", "sarcasm_pred", "sarcasm_threshold"]],
        on="row_id", how="left"
    )
    return merged

def pick_sarcasm_label_index(id2label: dict) -> int:
    norm = {int(k): str(v).lower() for k, v in id2label.items()}
    for idx, name in norm.items():
        if "sarcas" in name:  # 'sarcasm' / 'sarcastic'
            return idx
    return 1 if 1 in norm else sorted(norm.keys())[-1]

def tokenize_batch(tokenizer, texts, max_length):
    # Токенизация в отдельном потоке (HF fast tokenizer — Rust, снимает GIL)
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

def save_checkpoint(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

def main():
    global STOP_REQUESTED

    device = detect_device()
    print(f"Using device: {device}")

    # Данные + резюмирование
    base_df = load_data(INPUT_PATH, TEXT_COLUMN)
    df = resume_merge(base_df, OUTPUT_PATH)
    texts = df[TEXT_COLUMN].fillna("").astype(str).tolist()

    # Загружаем модель/токенайзер
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    id2label = model.config.id2label or {i: f"LABEL_{i}" for i in range(model.config.num_labels)}
    sarcasm_idx = pick_sarcasm_label_index(id2label)
    print(f"Sarcasm class index: {sarcasm_idx} ({id2label.get(sarcasm_idx, 'unknown')})")

    # Какие строки ещё не размечены?
    need_mask = df.get("sarcasm_score").isna() if "sarcasm_score" in df.columns else pd.Series(True, index=df.index)
    to_process_idx = df.index[need_mask].tolist()
    total_needed = len(to_process_idx)
    if total_needed == 0:
        print("Nothing to do: all rows already labeled.")
        return

    # Организуем списки микро-батчей индексов
    micro_batches = [
        to_process_idx[i:i + BATCH_SIZE]
        for i in range(0, total_needed, BATCH_SIZE)
    ]

    processed_since_save = 0
    pbar = tqdm(total=total_needed, desc="Annotating", unit="rows")

    try:
        # Пул токенизаций
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            # Будем слать задания токенизации пачками PREFETCH
            submit_ptr = 0
            futures = []

            def submit_more():
                nonlocal submit_ptr
                while submit_ptr < len(micro_batches) and len(futures) < PREFETCH:
                    mb_idx = micro_batches[submit_ptr]
                    mb_texts = [texts[i] for i in mb_idx]
                    fut = ex.submit(tokenize_batch, tokenizer, mb_texts, MAX_LENGTH)
                    futures.append((mb_idx, fut))
                    submit_ptr += 1

            # стартовое наполнение очереди
            submit_more()

            while futures:
                if STOP_REQUESTED:
                    print("\nStop requested. Saving checkpoint...")
                    break

                # берём первую готовую токенизацию
                done_any = False
                for k, (mb_idx, fut) in enumerate(futures):
                    if fut.done():
                        enc = fut.result()
                        # освобождаем слот и тут же подаём новое задание
                        futures.pop(k)
                        submit_more()
                        done_any = True

                        # инференс (последовательно и аккуратно на одном устройстве)
                        with torch.inference_mode():
                            enc = {k2: v.to(device) for k2, v in enc.items()}
                            logits = model(**enc).logits
                            probs = torch.softmax(logits, dim=1)[:, sarcasm_idx].detach().cpu().numpy()

                        # записываем результаты
                        df.loc[mb_idx, "sarcasm_score"] = probs
                        df.loc[mb_idx, "sarcasm_pred"] = (probs >= THRESHOLD).astype(int)
                        df.loc[mb_idx, "sarcasm_threshold"] = THRESHOLD

                        processed_since_save += len(mb_idx)
                        pbar.update(len(mb_idx))

                        # периодическое сохранение
                        if processed_since_save >= SAVE_EVERY:
                            save_checkpoint(df, OUTPUT_PATH)
                            processed_since_save = 0
                        break  # выходим из for, чтобы пересобрать futures и продолжить
                if not done_any:
                    # если ничего не готово, подождём чуть-чуть
                    # (без busy-wait, но tqdm и так обновляется)
                    pass

        # финальный сейв (или сейв при мягкой остановке)
        save_checkpoint(df, OUTPUT_PATH)
        print(f"Saved results to: {OUTPUT_PATH}")

    except Exception as e:
        # В случае исключения — тоже сохраняем прогресс
        print(f"\nException: {e}\nSaving partial results to: {OUTPUT_PATH}")
        save_checkpoint(df, OUTPUT_PATH)
        raise
    finally:
        pbar.close()

if __name__ == "__main__":
    main()
