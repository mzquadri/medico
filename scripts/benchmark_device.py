"""What one training step costs, on each device, under the current settings.

The hyperparameters in train_phase4_optimized.py are labelled "optimized for Phase 4 +
Intel Arc 2GB VRAM": batch size 4, gradient checkpointing on whenever the device is XPU,
and 32 accumulation steps to hold the effective batch at 128. Those are reasonable choices
for a 2 GB card. On a larger one they are not obviously right, and nothing in the
repository measured the difference.

This measures it. Four configurations on every available device: the current settings, the
same batch without checkpointing, a larger batch without it, and bfloat16 autocast, which
the script never enables because AMP_ENABLED requires CUDA.

It reports throughput and peak device memory. It is not a training result and produces no
model: the input is generated noise of the documented shape, and nothing about accuracy can
be read from it. Changing batch size changes BatchNorm statistics, so a faster
configuration is not automatically the right one, and this script deliberately recommends
nothing.

    python scripts/benchmark_device.py [--steps 20]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The training script imports the medico package, so the repository root has to be
# importable. pytest puts it there; a plain `python scripts/...` does not.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_script():
    """Import the training script without running main(), as the tests do."""
    spec = importlib.util.spec_from_file_location("t4", ROOT / "train_phase4_optimized.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["t4"] = module
    spec.loader.exec_module(module)
    return module


def devices(torch) -> list[str]:
    found = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def describe(torch, device: str) -> str:
    if device == "xpu":
        return torch.xpu.get_device_properties(0).name
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def peak_memory_gb(torch, device: str) -> float | None:
    if device == "xpu":
        return torch.xpu.max_memory_allocated() / 1024**3
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**3
    return None


def reset_memory(torch, device: str) -> None:
    if device == "xpu":
        torch.xpu.reset_peak_memory_stats()
    elif device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def synchronize(torch, device: str) -> None:
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def one_configuration(script, torch, device, *, batch, checkpointing, autocast, steps, warmup):
    """Time forward, loss and backward for a configuration. Returns None if it will not fit."""
    model = script.DenseNet121(
        num_classes=len(script.SELECTED_LABELS),
        grayscale=True,
        use_gradient_checkpointing=checkpointing,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    images = torch.randn(batch, 1, script.IMG_SIZE, script.IMG_SIZE, device=device)
    targets = torch.randint(0, 2, (batch, len(script.SELECTED_LABELS)),
                            device=device).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def step():
        optimizer.zero_grad(set_to_none=True)
        if autocast:
            with torch.amp.autocast(device, dtype=torch.bfloat16):
                loss = loss_fn(model(images), targets)
        else:
            loss = loss_fn(model(images), targets)
        loss.backward()
        optimizer.step()

    try:
        reset_memory(torch, device)
        for _ in range(warmup):
            step()
        synchronize(torch, device)
        timings = []
        for _ in range(steps):
            started = time.perf_counter()
            step()
            synchronize(torch, device)
            timings.append(time.perf_counter() - started)
    except RuntimeError as exc:                     # out of memory, most likely
        return {"error": type(exc).__name__, "detail": str(exc)[:80]}

    median = statistics.median(timings)
    return {
        "batch": batch,
        "checkpointing": checkpointing,
        "autocast_bf16": autocast,
        "step_ms": round(median * 1000, 1),
        "images_per_second": round(batch / median, 1),
        "peak_memory_gb": (round(peak_memory_gb(torch, device), 2)
                           if peak_memory_gb(torch, device) is not None else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import torch

    script = load_script()
    print(f"\nmodel   DenseNet-121, grayscale stem, {len(script.SELECTED_LABELS)} outputs")
    print(f"input   {script.IMG_SIZE}x{script.IMG_SIZE}, generated noise, not patient data")
    print(f"script  BATCH_SIZE={script.BATCH_SIZE}, "
          f"USE_GRADIENT_CHECKPOINTING={script.USE_GRADIENT_CHECKPOINTING}, "
          f"ACCUMULATION_STEPS={script.ACCUMULATION_STEPS}\n")

    plans = [
        ("current settings", script.BATCH_SIZE, True, False),
        ("no checkpointing", script.BATCH_SIZE, False, False),
        ("batch 32, no checkpointing", 32, False, False),
        ("batch 32, bfloat16 autocast", 32, False, True),
    ]
    record: dict[str, dict] = {}
    for device in devices(torch):
        print(f"  {describe(torch, device)} ({device})")
        record[device] = {"hardware": describe(torch, device), "configurations": {}}
        for label, batch, checkpointing, autocast in plans:
            if autocast and device == "cpu":
                continue                            # bfloat16 autocast on CPU is a different question
            result = one_configuration(script, torch, device, batch=batch,
                                       checkpointing=checkpointing, autocast=autocast,
                                       steps=args.steps, warmup=args.warmup)
            record[device]["configurations"][label] = result
            if "error" in result:
                print(f"    {label:30s} {result['error']}: {result['detail']}")
            else:
                memory = (f"{result['peak_memory_gb']:.2f} GB"
                          if result["peak_memory_gb"] is not None else "n/a")
                print(f"    {label:30s} {result['step_ms']:8.1f} ms/step "
                      f"{result['images_per_second']:8.1f} img/s   peak {memory}")
        print()

    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
