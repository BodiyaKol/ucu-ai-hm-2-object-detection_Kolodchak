"""FPS measurement (same GPU, same imgsz, batch=1, warmup excluded)."""
import time


def measure_fps(predict_fn, images, warmup: int = 10) -> float:
    for img in images[:warmup]:
        predict_fn(img)
    # TODO: torch.cuda.synchronize() before/after if on GPU
    start = time.perf_counter()
    for img in images[warmup:]:
        predict_fn(img)
    elapsed = time.perf_counter() - start
    return (len(images) - warmup) / elapsed
