"""The GPU must be used when it is there, whether or not IPEX is installed.

The XPU branch used to sit inside `try: import intel_extension_for_pytorch`, so the
availability question was only ever asked when IPEX imported successfully. On a machine
with a working Arc GPU and no IPEX the import raised, the check never ran, and the script
fell through DirectML to CPU while printing a single line saying so.

That was not a hypothetical. torch has carried native XPU support since 2.5,
requirements.txt documents a 2.11.0+xpu build and does not list IPEX, and IPEX was not
installed on the development machine, so the documented environment could never take the
XPU path. Measured on an Arc Pro 140T afterwards: 19.5 images per second on XPU against
4.8 on CPU at the script's own batch size, so the cost was roughly four times the
training time, silently.

These tests re-import the script with the device situation forced, because the decision is
made once at import time.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def import_script(name: str = "t4_device"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "train_phase4_optimized.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def without_ipex(monkeypatch):
    """Make `import intel_extension_for_pytorch` fail, as on a plain xpu torch install."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in {"intel_extension_for_pytorch", "torch_directml"}:
            raise ImportError(f"no module named {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


@pytest.fixture
def pretend_xpu_is_available(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda index=0: "Test Arc Device")


def test_an_available_xpu_is_used_even_without_ipex(
    without_ipex, pretend_xpu_is_available, monkeypatch
):
    """The regression. IPEX absent, XPU present, so XPU must win."""
    module = import_script("t4_xpu_no_ipex")
    assert str(module.DEVICE) == "xpu"
    assert module.IS_DIRECTML is False


def test_gradient_checkpointing_follows_the_xpu_decision(
    without_ipex, pretend_xpu_is_available
):
    """It is keyed on the device, so a wrong device silently changes memory behaviour."""
    module = import_script("t4_xpu_ckpt")
    assert module.USE_GRADIENT_CHECKPOINTING is True


def test_cpu_is_used_when_no_accelerator_is_present(without_ipex, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    module = import_script("t4_cpu")
    assert str(module.DEVICE) == "cpu"
    assert module.IS_DIRECTML is False
    assert module.USE_GRADIENT_CHECKPOINTING is False


def test_cuda_is_still_preferred_over_cpu(without_ipex, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    module = import_script("t4_cuda")
    assert str(module.DEVICE) == "cuda"


def test_the_availability_check_is_not_gated_on_an_optional_import():
    """Read as code, so the shape of the bug cannot return.

    An `import intel_extension_for_pytorch` placed before the availability question puts
    the GPU behind a package the documented requirements do not install. Comments are
    stripped first: this file and the script both discuss the import in prose, and a
    guard that trips on its own explanation is worse than no guard.
    """
    source = (ROOT / "train_phase4_optimized.py").read_text(encoding="utf-8")
    marker = "# Device Setup - Intel Arc GPU Support"
    block = source[source.index(marker):source.index(marker) + 2500]
    code = "\n".join(
        line for line in block.splitlines() if not line.lstrip().startswith("#")
    )
    availability = code.index("torch.xpu.is_available()")
    ipex = code.find("import intel_extension_for_pytorch")
    assert ipex == -1 or ipex > availability, (
        "the IPEX import must not come before the XPU availability check"
    )
