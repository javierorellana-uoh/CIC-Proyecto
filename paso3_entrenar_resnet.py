#!/usr/bin/env python
"""PASO 3: crear el ResNet-50 nuevo ajustado con CheXpert small.

Parte de los pesos de `resnet50-res512-all` (no desde cero) y lo ajusta con las
imágenes frontales de CheXpert small a la resolución indicada (320 px; 224 px
como control). Separa por PACIENTE un 10 % de train para validación, guarda el
mejor modelo según el AUC de validación y calcula sus umbrales de decisión
(índice de Youden, el mismo criterio que usa TorchXRayVision).

Todo queda en proyecto\\modelos\\<nombre>\\:
    pesos.pt        pesos del modelo nuevo
    config.json     resolución, semilla, hiperparámetros, versiones, umbrales, mejor época
    historial.csv   pérdida y AUC de validación por época

Uso (desde la raíz del repo):
    python proyecto\\paso3_entrenar_resnet.py --chex "RUTA\\Chexpert small" --res 320
    python proyecto\\paso3_entrenar_resnet.py --chex "RUTA\\Chexpert small" --res 224
Prueba rápida (minutos) para comprobar que todo funciona:
    python proyecto\\paso3_entrenar_resnet.py --chex "RUTA\\Chexpert small" --res 320 --max-imgs 2000 --epocas 1
"""
import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun_proyecto as cp  # noqa: E402
xrv = cp.xrv


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cp.RAIZ,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def construir_datos(a):
    """CheXpert small, solo frontales, etiquetas alineadas a las 18 salidas del modelo."""
    aumento = torchvision.transforms.Compose([
        xrv.datasets.ToPILImage(),
        torchvision.transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        torchvision.transforms.ToTensor(),
    ])
    tf = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                         xrv.datasets.XRayResizer(a.res, engine="cv2")])
    with cp.bench.quiet():
        d = xrv.datasets.CheX_Dataset(imgpath=a.chex, csvpath=os.path.join(a.chex, "train.csv"),
                                      views=["PA", "AP"], unique_patients=False, transform=tf)
        xrv.datasets.relabel_dataset(xrv.models.ResNet.targets, d, silent=True)
    d_aug = xrv.datasets.CheX_Dataset(imgpath=a.chex, csvpath=os.path.join(a.chex, "train.csv"),
                                      views=["PA", "AP"], unique_patients=False, transform=tf,
                                      data_aug=aumento)
    with cp.bench.quiet():
        xrv.datasets.relabel_dataset(xrv.models.ResNet.targets, d_aug, silent=True)

    # partición por paciente: ningún paciente queda en entrenamiento y validación a la vez
    pacientes = d.csv["patientid"].values
    unicos = np.unique(pacientes)
    rng = np.random.default_rng(a.semilla)
    val_pac = set(rng.choice(unicos, int(len(unicos) * a.val_frac), replace=False))
    es_val = np.array([p in val_pac for p in pacientes])
    idx_tr, idx_val = np.where(~es_val)[0], np.where(es_val)[0]
    if a.max_imgs:
        idx_tr = rng.choice(idx_tr, min(a.max_imgs, len(idx_tr)), replace=False)
        idx_val = rng.choice(idx_val, min(max(a.max_imgs // 5, 200), len(idx_val)), replace=False)
    tr = xrv.datasets.SubsetDataset(d_aug, np.sort(idx_tr))   # con aumento
    va = xrv.datasets.SubsetDataset(d, np.sort(idx_val))      # sin aumento
    return tr, va, len(set(pacientes[idx_tr])), len(set(pacientes[idx_val]))


def construir_modelo(a, device):
    with cp.bench.quiet():
        m = xrv.models.ResNet(weights="resnet50-res512-all")
    cp.bench.zero_denormals(m)
    m.op_threshs = None          # entrenar con logits crudos
    m.apply_sigmoid = False
    m.input_resolution = a.res   # el modelo trabaja nativamente a la nueva resolución
    return m.to(device).to(memory_format=torch.channels_last)


def perdida(logits, y):
    """BCE solo sobre etiquetas conocidas (NaN = desconocida o incierta)."""
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits[mask], y[mask])


@torch.no_grad()
def validar(m, dl, device, amp):
    m.eval()
    logits, labels = [], []
    for b in dl:
        x = b["img"].to(device).float().contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits.append(m(x).float().cpu())
        labels.append(b["lab"])
    L, Y = torch.cat(logits).numpy(), torch.cat(labels).numpy()
    P = 1 / (1 + np.exp(-L))
    aucs, umbrales = {}, []
    for j, t in enumerate(m.targets):
        y = Y[:, j]
        ok = ~np.isnan(y)
        if t and len(np.unique(y[ok])) == 2:
            aucs[t] = roc_auc_score(y[ok], P[ok, j])
            fpr, tpr, th = roc_curve(y[ok], P[ok, j])
            umbrales.append(float(th[np.argmax(tpr - fpr)]))    # Youden, como TorchXRayVision
        else:
            umbrales.append(float("nan"))
    comunes = [aucs[p] for p in cp.COMUNES if p in aucs]
    return float(np.mean(comunes)) if comunes else float("nan"), aucs, umbrales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chex", required=True, help="carpeta CheXpert-v1.0-small (con train.csv)")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--paciencia", type=int, default=2, help="épocas sin mejorar antes de detener")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-imgs", type=int, default=0, help="limitar imágenes de entrenamiento (prueba rápida)")
    ap.add_argument("--nombre", default=None, help="por defecto resnet50-res<res>-chex-s<semilla>")
    a = ap.parse_args()

    fijar_semilla(a.semilla)
    device = cp.bench.pick_device("auto")
    amp = str(device).startswith("cuda")
    if amp:
        torch.backends.cudnn.benchmark = True
    nombre = a.nombre or f"resnet50-res{a.res}-chex-s{a.semilla}" + ("-prueba" if a.max_imgs else "")
    out = os.path.join(cp.AQUI, "modelos", nombre)
    os.makedirs(out, exist_ok=True)

    tr, va, n_pac_tr, n_pac_va = construir_datos(a)
    print(f"Entrenamiento: {len(tr)} imágenes ({n_pac_tr} pacientes) | validación: {len(va)} ({n_pac_va} pacientes)")
    print(f"Resolución {a.res} px | dispositivo {device} | salida {out}", flush=True)

    g = torch.Generator().manual_seed(a.semilla)
    dl_tr = torch.utils.data.DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                                        pin_memory=amp, drop_last=True, generator=g,
                                        persistent_workers=a.workers > 0)
    dl_va = torch.utils.data.DataLoader(va, batch_size=a.bs * 2, num_workers=a.workers, pin_memory=amp)

    m = construir_modelo(a, device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epocas * len(dl_tr))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    auc0, _, _ = validar(m, dl_va, device, amp)
    print(f"AUC de validación antes de entrenar (modelo original a {a.res} px): {auc0:.4f}", flush=True)

    config = {"nombre": nombre, "base": "resnet50-res512-all", "input_resolution": a.res,
              "targets": list(m.targets), "args": vars(a),
              "n_train": len(tr), "n_val": len(va), "pacientes_train": n_pac_tr, "pacientes_val": n_pac_va,
              "auc_val_inicial": auc0, "entorno": cp.bench.env_info(device), "git_commit": git_commit(),
              "python": platform.python_version()}
    hist_path = os.path.join(out, "historial.csv")
    with open(hist_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoca", "perdida_train", "auc_val_7comunes", "segundos"])

    mejor, sin_mejora = -1.0, 0
    for ep in range(1, a.epocas + 1):
        m.train()
        t0, suma, n = time.time(), 0.0, 0
        for i, b in enumerate(dl_tr, 1):
            x = b["img"].to(device, non_blocking=True).float().contiguous(memory_format=torch.channels_last)
            y = b["lab"].to(device, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = m(x)
            loss = perdida(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            suma += loss.item(); n += 1
            if i % 200 == 0:
                print(f"  época {ep} | lote {i}/{len(dl_tr)} | pérdida {suma / n:.4f} | {time.time() - t0:.0f} s", flush=True)
        auc, aucs, umbrales = validar(m, dl_va, device, amp)
        seg = time.time() - t0
        print(f"Época {ep}: pérdida {suma / n:.4f} | AUC val (7 comunes) {auc:.4f} | {seg / 60:.1f} min", flush=True)
        with open(hist_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(suma / n, 5), round(auc, 5), round(seg, 1)])
        if auc > mejor:
            mejor, sin_mejora = auc, 0
            torch.save(m.model.state_dict(), os.path.join(out, "pesos.pt"))
            config.update({"mejor_epoca": ep, "auc_val": auc, "auc_val_por_patologia": aucs,
                           "op_threshs": umbrales})
            with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  -> nuevo mejor modelo guardado", flush=True)
        else:
            sin_mejora += 1
            if sin_mejora >= a.paciencia:
                print(f"Sin mejora en {a.paciencia} épocas: se detiene.")
                break

    print(f"\nListo. Mejor AUC de validación: {mejor:.4f} (antes de entrenar: {auc0:.4f})")
    print(f"Modelo en: {out}")
    print("Para evaluarlo con el mismo protocolo que la línea base:")
    print(f'  python proyecto\\paso2_evaluar_base.py --nih "RUTA_NIH" --chex "{a.chex}" '
          f'--modelos "{out}" --agregar')


if __name__ == "__main__":
    main()
