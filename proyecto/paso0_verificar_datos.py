#!/usr/bin/env python
"""PASO 0: verificar que NIH ChestX-ray14 y CheXpert small están completos y
en el formato que espera TorchXRayVision.

Uso (desde la raíz del repo):
    python proyecto\\paso0_verificar_datos.py --nih C:\\datos\\NIH --chex C:\\datos\\CheXpert-v1.0-small

Si NIH viene de Kaggle (imágenes repartidas en images_001\\images ... images_012\\images),
agregue --organizar-nih para moverlas todas a una sola carpeta NIH\\images.
"""
import argparse
import glob
import os
import random
import shutil
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comun_proyecto import RAIZ  # noqa: E402

OK, MAL, AVISO = "[OK]   ", "[ERROR]", "[AVISO]"


def tamanos(archivos, n=200):
    muestra = random.Random(0).sample(archivos, min(n, len(archivos)))
    return pd.Series([Image.open(f).size for f in muestra]).value_counts()


def verificar_nih(carpeta, organizar):
    print("\n========== NIH ChestX-ray14 ==========")
    destino = os.path.join(carpeta, "images")
    anidadas = sorted(glob.glob(os.path.join(carpeta, "images_*", "images")))
    if anidadas:
        n_anid = sum(len(glob.glob(os.path.join(a, "*.png"))) for a in anidadas)
        print(f"{AVISO} Formato Kaggle: {n_anid} imágenes repartidas en {len(anidadas)} subcarpetas.")
        if organizar:
            os.makedirs(destino, exist_ok=True)
            movidas = 0
            for a in anidadas:
                for f in glob.glob(os.path.join(a, "*.png")):
                    dst = os.path.join(destino, os.path.basename(f))
                    if not os.path.exists(dst):
                        shutil.move(f, dst)
                        movidas += 1
            print(f"{OK} Movidas {movidas} imágenes a {destino}")
        else:
            print("        Ejecute de nuevo con --organizar-nih para juntarlas en una sola carpeta.")
    if not os.path.isdir(destino):
        print(f"{MAL} No existe {destino}. Debe contener todas las imágenes .png de NIH.")
        return
    pngs = glob.glob(os.path.join(destino, "*.png"))
    print(f"{OK if len(pngs) == 112120 else AVISO} {len(pngs)} imágenes en {destino} (esperadas: 112.120)")
    if pngs:
        print("        Tamaños (muestra de 200):", {f"{k[0]}x{k[1]}": int(v) for k, v in tamanos(pngs).items()})

    csv = pd.read_csv(os.path.join(RAIZ, "torchxrayvision", "data", "Data_Entry_2017_v2020.csv.gz"))
    faltan = set(csv["Image Index"]) - {os.path.basename(p) for p in pngs}
    print(f"{OK if not faltan else MAL} Imágenes del CSV de la librería que faltan: {len(faltan)}")

    for nombre in ["train_val_list.txt", "test_list.txt"]:
        enc = glob.glob(os.path.join(carpeta, "**", nombre), recursive=True)
        if enc:
            n = sum(1 for l in open(enc[0]) if l.strip())
            print(f"{OK} {nombre}: {n} imágenes ({enc[0]})")
        else:
            print(f"{MAL} No se encontró {nombre}. Viene con el dataset (Kaggle o NIH); cópielo a {carpeta}")


def verificar_chexpert(carpeta):
    print("\n========== CheXpert small ==========")
    for nombre in ["train.csv", "valid.csv"]:
        p = os.path.join(carpeta, nombre)
        if not os.path.exists(p):
            print(f"{MAL} No existe {p}. --chex debe apuntar a la carpeta CheXpert-v1.0-small")
            return
    tr = pd.read_csv(os.path.join(carpeta, "train.csv"))
    va = pd.read_csv(os.path.join(carpeta, "valid.csv"))
    print(f"{OK if len(tr) == 223414 else AVISO} train.csv: {len(tr)} filas (esperadas: 223.414)")
    print(f"{OK if len(va) == 234 else AVISO} valid.csv: {len(va)} filas (esperadas: 234)")

    lib = pd.read_csv(os.path.join(RAIZ, "torchxrayvision", "data", "chexpert_train.csv.gz"))
    iguales = set(tr.Path) == set(lib.Path)
    print(f"{OK if iguales else AVISO} train.csv {'coincide' if iguales else 'NO coincide'} con el CSV de la librería")

    def ruta(p):
        return os.path.join(carpeta, p.replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", ""))
    for nombre, df in [("valid", va), ("train", tr.sample(min(2000, len(tr)), random_state=0))]:
        faltan = [p for p in df.Path if not os.path.exists(ruta(p))]
        etiqueta = "todas" if nombre == "valid" else "muestra de 2000"
        print(f"{OK if not faltan else MAL} Imágenes de {nombre} que faltan ({etiqueta}): {len(faltan)}")
    existentes = [ruta(p) for p in tr.Path.sample(min(500, len(tr)), random_state=1) if os.path.exists(ruta(p))]
    if existentes:
        lados = pd.Series([min(Image.open(f).size) for f in existentes[:200]]).value_counts()
        print("        Lado menor de las imágenes (muestra de 200):", {int(k): int(v) for k, v in lados.head(5).items()})
    frontales = tr["Frontal/Lateral"].eq("Frontal").sum()
    print(f"        Imágenes frontales en train: {frontales}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", help="carpeta raíz de NIH (la que contiene images\\ o images_001\\...)")
    ap.add_argument("--chex", help="carpeta CheXpert-v1.0-small")
    ap.add_argument("--organizar-nih", action="store_true")
    a = ap.parse_args()
    if a.nih:
        verificar_nih(a.nih, a.organizar_nih)
    if a.chex:
        verificar_chexpert(a.chex)
    if not (a.nih or a.chex):
        ap.print_help()


if __name__ == "__main__":
    main()
