#!/usr/bin/env python
"""PASO 2b: técnicas para recuperar precisión con imágenes de 320 px SIN entrenar.

Sobre el modelo original (resnet50-res512-all por defecto) compara, en NIH test
reducido a 320 px y en CheXpert small valid:

  * interpolación   : cómo se agranda la imagen de 320 a 512 px antes del modelo
                      (interna bilineal = línea base, vecino más cercano, bicúbica, Lanczos)
  * AdaBN           : recalcular las estadísticas de BatchNorm con imágenes de 320 px
                      del dominio de destino, sin etiquetas (Li et al., 2016)
  * TTA             : promediar las predicciones de 5 variantes de la imagen
                      (original, zoom 5 %, desplazamientos de 3 %; sin espejo)
  * AdaBN + TTA     : ambas
  * recalibración   : nuevos umbrales de decisión (índice de Youden) calculados con
                      imágenes de 320 px que NO están en la prueba. No cambia el AUC;
                      corrige sensibilidad y especificidad.

Brecha recuperada (NIH) = (AUC técnica - AUC 320 base) / (AUC original - AUC 320 base).

Uso (desde la raíz del repo):
    python proyecto\\paso2b_tecnicas_sin_entrenar.py --nih "RUTA_NIH" --chex "RUTA\\Chexpert small"
Prueba rápida:
    ... --max-imgs 1000 --n-boot 100 --n-adapt 1000
"""
import argparse
import copy
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun_proyecto as cp  # noqa: E402
xrv = cp.xrv

METODOS = {"vecino": cv2.INTER_NEAREST, "bicubica": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


class Agrandar:
    """Agranda la imagen [1,H,W] a res x res con un método de OpenCV."""
    def __init__(self, res, metodo):
        self.res, self.metodo = res, metodo

    def __call__(self, img):
        return cv2.resize(img[0], (self.res, self.res), interpolation=self.metodo)[None].astype(np.float32)


def tf(*pasos):
    return torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), *pasos])


def nih(carpeta, lista, *pasos):
    d = xrv.datasets.NIH_Dataset(imgpath=carpeta, views=["PA", "AP"], unique_patients=False, transform=tf(*pasos))
    return xrv.datasets.SubsetDataset(d, np.where(d.csv["Image Index"].isin(lista).values)[0])


def chex(carpeta, csv, *pasos):
    return xrv.datasets.CheX_Dataset(imgpath=carpeta, csvpath=os.path.join(carpeta, csv),
                                     views=["PA", "AP"], unique_patients=False, transform=tf(*pasos))


@torch.no_grad()
def predecir_tta(modelo, d, device, bs, workers):
    """Promedio de 5 variantes geométricas suaves (sin espejo: el corazón está a la izquierda)."""
    variantes = [(1.0, 0, 0), (1.05, 0, 0), (1.0, 0.03, 0), (1.0, -0.03, 0), (1.0, 0, 0.03)]
    dl = torch.utils.data.DataLoader(d, batch_size=bs, num_workers=workers)
    out = []
    with cp.bench.quiet():
        for b in dl:
            x = b["img"].to(device).float()
            acc = 0
            for s, tx, ty in variantes:
                theta = torch.tensor([[1 / s, 0, tx], [0, 1 / s, ty]], dtype=x.dtype, device=device)
                grid = F.affine_grid(theta.expand(len(x), 2, 3), x.shape, align_corners=False)
                xv = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
                with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                    acc = acc + modelo(xv).float()
            out.append((acc / len(variantes)).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def adabn(modelo, d, device, bs, workers):
    """Copia del modelo con las estadísticas de BatchNorm recalculadas sobre d (sin etiquetas)."""
    m = copy.deepcopy(modelo)
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.reset_running_stats()
            mod.momentum = None            # promedio acumulado exacto
    m.train()
    dl = torch.utils.data.DataLoader(d, batch_size=bs, num_workers=workers, shuffle=True,
                                     generator=torch.Generator().manual_seed(0))
    with cp.bench.quiet():
        for b in dl:
            m(b["img"].to(device).float())
    return m.eval()


def umbrales_youden(probs, labels, pathologies, targets):
    """Umbral por salida que maximiza sensibilidad + especificidad (criterio de TorchXRayVision)."""
    u = []
    for t in targets:
        if t and t in pathologies:
            y = labels[:, pathologies.index(t)]
            ok = ~np.isnan(y)
            if len(np.unique(y[ok])) == 2:
                fpr, tpr, th = roc_curve(y[ok], probs[ok, targets.index(t)])
                u.append(float(np.clip(th[np.argmax(tpr - fpr)], 1e-6, 1 - 1e-6)))
                continue
        u.append(float("nan"))
    return torch.tensor(u, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", required=True)
    ap.add_argument("--chex", required=True)
    ap.add_argument("--modelo", default="resnet50-res512-all")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-adapt", type=int, default=5000,
                    help="imágenes (fuera de la prueba) para AdaBN y para recalibrar umbrales")
    ap.add_argument("--max-imgs", type=int, default=0, help="limitar NIH test (prueba rápida)")
    ap.add_argument("--out", default=os.path.join(cp.AQUI, "resultados", "paso2b"))
    a = ap.parse_args()
    device = cp.bench.pick_device("auto")
    rng = np.random.default_rng(0)

    lista_test = cp.leer_lista(glob.glob(os.path.join(a.nih, "**", "test_list.txt"), recursive=True)[0])
    lista_tv = sorted(cp.leer_lista(glob.glob(os.path.join(a.nih, "**", "train_val_list.txt"), recursive=True)[0]))
    lista_adapt = set(rng.choice(lista_tv, min(a.n_adapt, len(lista_tv)), replace=False))
    dir_orig, dir320 = os.path.join(a.nih, "images"), os.path.join(a.nih, "images_320")
    if not os.path.isdir(dir320):
        sys.exit("Falta NIH\\images_320: ejecute antes el paso 1.")
    lim = lambda d: cp.limitar(d, a.max_imgs)

    # ---------------- conjuntos (mismas imágenes en todas las variantes) ----------------
    nih_sets = {
        "original (512)": lim(nih(dir_orig, lista_test, xrv.datasets.XRayResizer(512, engine="cv2"))),
        "320 base (bilineal interna)": lim(nih(dir320, lista_test)),
        **{f"320 interp. {k}": lim(nih(dir320, lista_test, Agrandar(512, v))) for k, v in METODOS.items()},
    }
    chex_sets = {
        "320 base (bilineal interna)": chex(a.chex, "valid.csv"),
        **{f"320 interp. {k}": chex(a.chex, "valid.csv", Agrandar(512, v)) for k, v in METODOS.items()},
    }
    nih_adapt = nih(dir320, lista_adapt)                                  # NIH train_val, no está en la prueba
    chex_adapt = cp.limitar(chex(a.chex, "train.csv", xrv.datasets.XRayResizer(320, engine="cv2")), a.n_adapt)
    n = {k: len(v) for k, v in nih_sets.items()}
    print(f"NIH test: {n['original (512)']} | CheXpert valid: {len(chex_sets['320 base (bilineal interna)'])} | "
          f"adaptación: NIH {len(nih_adapt)}, CheXpert {len(chex_adapt)} | dispositivo {device}", flush=True)

    modelo, targets, umbrales = cp.cargar_modelo(a.modelo, device)
    R = {"modelo": a.modelo, "device": str(device), "env": cp.bench.env_info(device),
         "n_imagenes": {"nih_test": n["original (512)"], "chex_valid": len(chex_sets["320 base (bilineal interna)"]),
                        "adapt_nih": len(nih_adapt), "adapt_chex": len(chex_adapt)},
         "nih": {}, "chex": {}}

    def registrar(dom, nombre, d, probs, u=umbrales, ref=None):
        dec = cp.decisiones(probs, u)
        met = cp.metricas(d.labels, list(d.pathologies), probs, dec, targets, d.csv["patientid"].values, a.n_boot)
        fila = {"metricas": met}
        if ref is not None:
            fila["fidelidad_vs_original"] = cp.fidelidad(ref, dec, targets)
        R[dom][nombre] = fila
        sens = np.mean([v["sensibilidad"] for k, v in met.items() if k != "PROMEDIO"])
        print(f"  [{dom}] {nombre:32s} AUC {met['PROMEDIO']['auc']:.4f} | sensibilidad media {sens:.2f}", flush=True)
        cp.guardar_json(R, os.path.join(a.out, "tecnicas.json"))
        return dec

    print("\n== Interpolación", flush=True)
    p = {}
    for nombre, d in nih_sets.items():
        p[nombre], _ = cp.predecir(modelo, d, device, a.bs, a.workers)
    ref = cp.decisiones(p["original (512)"], umbrales)
    for nombre, d in nih_sets.items():
        registrar("nih", nombre, d, p[nombre], ref=ref)
    pc = {}
    for nombre, d in chex_sets.items():
        pc[nombre], _ = cp.predecir(modelo, d, device, a.bs, a.workers)
        registrar("chex", nombre, d, pc[nombre])

    base_nih, base_chex = nih_sets["320 base (bilineal interna)"], chex_sets["320 base (bilineal interna)"]

    print("\n== TTA (promedio de 5 variantes)", flush=True)
    registrar("nih", "320 TTA", base_nih, predecir_tta(modelo, base_nih, device, a.bs, a.workers), ref=ref)
    registrar("chex", "320 TTA", base_chex, predecir_tta(modelo, base_chex, device, a.bs, a.workers))

    print("\n== AdaBN (estadísticas de BatchNorm del dominio de destino, sin etiquetas)", flush=True)
    m_nih = adabn(modelo, nih_adapt, device, a.bs, a.workers)
    registrar("nih", "320 AdaBN", base_nih, cp.predecir(m_nih, base_nih, device, a.bs, a.workers)[0], ref=ref)
    registrar("nih", "320 AdaBN + TTA", base_nih, predecir_tta(m_nih, base_nih, device, a.bs, a.workers), ref=ref)
    m_chex = adabn(modelo, chex_adapt, device, a.bs, a.workers)
    registrar("chex", "320 AdaBN", base_chex, cp.predecir(m_chex, base_chex, device, a.bs, a.workers)[0])
    registrar("chex", "320 AdaBN + TTA", base_chex, predecir_tta(m_chex, base_chex, device, a.bs, a.workers))
    del m_nih, m_chex

    print("\n== Recalibración de umbrales (Youden con imágenes fuera de la prueba)", flush=True)
    pa, _ = cp.predecir(modelo, nih_adapt, device, a.bs, a.workers)
    u_nih = umbrales_youden(pa, nih_adapt.labels, list(nih_adapt.pathologies), targets)
    registrar("nih", "320 umbrales recalibrados", base_nih, p["320 base (bilineal interna)"], u=u_nih, ref=ref)
    pa, _ = cp.predecir(modelo, chex_adapt, device, a.bs, a.workers)
    u_chex = umbrales_youden(pa, chex_adapt.labels, list(chex_adapt.pathologies), targets)
    registrar("chex", "320 umbrales recalibrados", base_chex, pc["320 base (bilineal interna)"], u=u_chex)

    # ---------------- brecha recuperada en NIH ----------------
    auc = lambda k: R["nih"][k]["metricas"]["PROMEDIO"]["auc"]
    techo, piso = auc("original (512)"), auc("320 base (bilineal interna)")
    R["brecha_nih"] = {"auc_original": techo, "auc_320_base": piso,
                       "recuperada_pct": {k: 100 * (auc(k) - piso) / (techo - piso) if techo != piso else float("nan")
                                          for k in R["nih"] if k != "original (512)"}}
    cp.guardar_json(R, os.path.join(a.out, "tecnicas.json"))

    print(f"\nBrecha NIH: original {techo:.4f} -> 320 base {piso:.4f} ({techo - piso:+.4f})")
    for k, v in R["brecha_nih"]["recuperada_pct"].items():
        print(f"  {k:32s} brecha recuperada: {v:6.1f} %")
    print("\nListo:", os.path.join(a.out, "tecnicas.json"), "\nAbra el notebook y ejecute Run All para ver los gráficos.")


if __name__ == "__main__":
    main()
