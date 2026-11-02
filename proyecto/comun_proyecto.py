"""Funciones compartidas por los scripts del proyecto
(destilación y modelos para radiografías de baja resolución).

Carga NIH ChestX-ray14 y CheXpert small con las clases de TorchXRayVision,
predice con los modelos de la librería y calcula las métricas del proyecto.
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torchvision
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
for p in (RAIZ, os.path.join(RAIZ, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

import torchxrayvision as xrv  # noqa: E402  (la librería de esta carpeta)
import common as bench  # noqa: E402  (benchmarks/common.py: carga de modelos, corrección de subnormales)

# Las 7 patologías que existen en NIH y en CheXpert
COMUNES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
           "Effusion", "Pneumonia", "Pneumothorax"]


# ----------------------------------------------------------------- datos
def leer_lista(path):
    with open(path) as f:
        return {l.strip() for l in f if l.strip()}


def _transform(res):
    pasos = [xrv.datasets.XRayCenterCrop()]
    if res:
        pasos.append(xrv.datasets.XRayResizer(res, engine="cv2"))
    return torchvision.transforms.Compose(pasos)


def dataset_nih(carpeta_imgs, lista=None, res=None):
    """NIH con todas las vistas frontales y todas las imágenes por paciente.
    lista: conjunto de nombres de archivo (p. ej. test_list.txt) para quedarse con esa partición.
    res: resolución a la que se reduce cada imagen (None = tal cual)."""
    d = xrv.datasets.NIH_Dataset(imgpath=carpeta_imgs, views=["PA", "AP"],
                                 unique_patients=False, transform=_transform(res))
    if lista is not None:
        idx = np.where(d.csv["Image Index"].isin(lista).values)[0]
        d = xrv.datasets.SubsetDataset(d, idx)
    return d


def dataset_chexpert(carpeta_small, csv="valid.csv", res=320):
    """CheXpert small (solo frontales). carpeta_small = carpeta CheXpert-v1.0-small."""
    return xrv.datasets.CheX_Dataset(imgpath=carpeta_small,
                                     csvpath=os.path.join(carpeta_small, csv),
                                     views=["PA", "AP"], unique_patients=False,
                                     transform=_transform(res))


def limitar(d, n, seed=0):
    """Submuestra aleatoria (para pruebas rápidas)."""
    if not n or n >= len(d):
        return d
    rng = np.random.default_rng(seed)
    return xrv.datasets.SubsetDataset(d, np.sort(rng.choice(len(d), n, replace=False)))


# --------------------------------------------------------------- modelos
def cargar_modelo(nombre, device):
    """Carga un modelo de la librería, corrige pesos subnormales y lo deja
    devolviendo probabilidades 'crudas' (sigmoide). Devuelve también los
    umbrales operativos originales para poder calcular decisiones."""
    with bench.quiet():
        m, res, kind, _ = bench.load_model(nombre, device)
    bench.zero_denormals(m)
    umbrales = getattr(m, "op_threshs", None)
    umbrales = None if umbrales is None else umbrales.detach().cpu().clone()
    if umbrales is not None:
        m.op_threshs = None
        m.apply_sigmoid = True
    return m, list(m.targets), umbrales


@torch.no_grad()
def predecir(modelo, dataset, device, bs=32, workers=4, amp=True):
    dl = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=workers,
                                     pin_memory=str(device).startswith("cuda"))
    salidas, t0 = [], time.perf_counter()
    usar_amp = amp and str(device).startswith("cuda")
    with bench.quiet():
        for lote in dl:
            x = lote["img"].to(device).float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=usar_amp):
                y = modelo(x)
            salidas.append(y.float().cpu())
    return torch.cat(salidas).numpy(), time.perf_counter() - t0


def decisiones(probs, umbrales):
    """Normaliza con los umbrales operativos del modelo: >0.5 = positivo."""
    if umbrales is None:
        return probs
    return xrv.models.op_norm(torch.from_numpy(probs), umbrales).numpy()


# --------------------------------------------------------------- métricas
def _grupos(pacientes):
    pac = np.asarray(pacientes)
    unicos, inv = np.unique(pac, return_inverse=True)
    orden = np.argsort(inv, kind="stable")
    cortes = np.cumsum(np.bincount(inv))[:-1]
    return np.split(orden, cortes)


def auc_ic(y, s, grupos, n_boot=500, seed=0):
    """AUC con IC95 % por bootstrap a nivel de paciente."""
    if len(np.unique(y)) < 2:
        return None
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def metricas(labels, pathologies, probs, decis, targets, pacientes, n_boot=500):
    """Métricas por patología común: AUC (IC95), AUPRC, sensibilidad,
    especificidad (en el umbral operativo del modelo) y Brier."""
    grupos = _grupos(pacientes)
    res = {}
    for p in COMUNES:
        if p not in pathologies or p not in targets or targets[targets.index(p)] == "":
            continue
        y = labels[:, pathologies.index(p)]
        ok = ~np.isnan(y)
        y, s, d = y[ok].astype(int), probs[ok, targets.index(p)], decis[ok, targets.index(p)]
        if len(np.unique(y)) < 2:
            continue
        g = grupos if ok.all() else _grupos(np.asarray(pacientes)[ok])
        auc = auc_ic(y, s, g, n_boot)
        pred = d > 0.5
        tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
        res[p] = {"auc": auc[0], "auc_ic95": [auc[1], auc[2]],
                  "auprc": float(average_precision_score(y, s)),
                  "prevalencia": float(y.mean()),
                  "sensibilidad": tp / max(tp + fn, 1),
                  "especificidad": tn / max(tn + fp, 1),
                  "brier": float(np.mean((s - y) ** 2)),
                  "n_pos": int(y.sum()), "n": int(len(y))}
    if res:
        res["PROMEDIO"] = {"auc": float(np.mean([v["auc"] for v in res.values()])),
                           "auprc": float(np.mean([v["auprc"] for v in res.values()]))}
    return res


def fidelidad(decis_a, decis_b, targets):
    """Cuánto cambian las respuestas entre dos versiones de las mismas imágenes."""
    cols = [targets.index(p) for p in COMUNES if p in targets]
    a, b = decis_a[:, cols], decis_b[:, cols]
    rho = [spearmanr(a[:, j], b[:, j]).statistic for j in range(a.shape[1])]
    return {"dif_media_prob": float(np.abs(a - b).mean()),
            "pct_decisiones_cambian": float(100 * ((a > 0.5) != (b > 0.5)).mean()),
            "spearman_medio": float(np.nanmean(rho))}


def guardar_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
