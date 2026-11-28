"""Sustituto local de `benchmarks/common.py` de TorchXRayVision.

El proyecto se escribió contra el `benchmarks/common.py` del repositorio original
(que no está incluido aquí). Este módulo implementa las cinco funciones que usan
los scripts del proyecto para poder ejecutarlos sin ese archivo externo:

    quiet, pick_device, env_info, zero_denormals, load_model

Si en el entorno existe un `common` real, `comun_proyecto.py` lo prefiere; este
módulo solo se usa como respaldo.
"""
import contextlib
import io
import os
import platform
import warnings

import torch
import torchxrayvision as xrv


@contextlib.contextmanager
def quiet():
    """Silencia warnings, stdout y stderr mientras se carga o infiere."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield


def pick_device(name="auto"):
    """Devuelve un torch.device a partir de 'auto', 'cuda', 'mps' o 'cpu'."""
    if name in (None, "", "auto"):
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)


def env_info(device):
    """Información del entorno con las mismas claves que los JSON del proyecto."""
    d = torch.device(device)
    en_cuda = d.type == "cuda" and torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchxrayvision": xrv.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_threads": os.cpu_count(),
        "device": d.type,
        "gpu": torch.cuda.get_device_name(0) if en_cuda else None,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def zero_denormals(model):
    """Pone a cero los valores subnormales (denormales) de parámetros y buffers.

    Los pesos de TorchXRayVision contienen algunos valores ~1e-38 que pueden
    ralentizar o desestabilizar la inferencia en algunos equipos. No afecta a
    valores normales (umbrales, pesos habituales).
    """
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            if t.is_floating_point():
                t.masked_fill_(t.abs() < torch.finfo(t.dtype).tiny, 0.0)
    return model


def load_model(name, device):
    """Carga un modelo de la librería y lo deja listo para predecir.

    Devuelve (modelo, resolución, tipo, None), igual que el `load_model`
    original. Conserva `op_threshs` para que `cargar_modelo` pueda capturar los
    umbrales operativos antes de pasar a sigmoide.
    """
    model = xrv.models.get_model(name, apply_sigmoid=True)
    model = model.to(device).eval()
    resolucion = getattr(model, "input_resolution", None)
    return model, resolucion, type(model).__name__, None
