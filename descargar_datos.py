#!/usr/bin/env python
"""Descarga un subconjunto mínimo de NIH ChestX-ray14 y CheXpert valid.

Pensado para poder ejecutar el paso 5 (efecto batch) sin bajar los ~42 GB de
NIH ni los ~11 GB de CheXpert. Deja los datos en la estructura que esperan los
scripts del proyecto:

    <destino>\\NIH\\images\\*.png
    <destino>\\NIH\\test_list.txt
    <destino>\\NIH\\train_val_list.txt
    <destino>\\CheXpert-v1.0-small\\valid.csv
    <destino>\\CheXpert-v1.0-small\\valid\\patient...\\study...\\view1_frontal.jpg

Fuentes:
  * NIH: https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset
  * CheXpert: https://huggingface.co/datasets/danjacobellis/chexpert
    (mirror de terceros; CheXpert requiere aceptar el acuerdo de Stanford).

Uso:
    python descargar_datos.py --destino C:\\datos --n-nih 234 --zips 001,006,012

Es idempotente: omite ficheros ya descargados/extraídos y reanuda descargas
interrumpidas mediante ficheros .part.
"""
import argparse
import json
import os
import random
import shutil
import time
import zipfile

import pandas as pd
import requests

HF_NIH = "https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data"
HF_CHEX = ("https://huggingface.co/datasets/danjacobellis/chexpert/resolve/main/data/"
           "validation-00000-of-00001.parquet")

# El parquet de HF guarda estas columnas como índices de clase; CheX_Dataset
# espera los valores originales de CheXpert.
MAPA_SEXO = {0: "Male", 1: "Female"}
MAPA_VISTA = {0: "Frontal", 1: "Lateral"}
MAPA_APPA = {0: "AP", 1: "PA", 2: "LL"}
MAPA_ETIQUETA = {0: None, 1: -1, 2: 0, 3: 1}
ETIQUETAS_CHEX = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
]


def descargar(url, destino, reanudar=True, timeout=(10, 120)):
    """Descarga a `destino` usando un .part reanudable."""
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    part = destino + ".part"
    offset = os.path.getsize(part) if reanudar and os.path.exists(part) else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        if r.status_code == 416:
            os.remove(part)
            return descargar(url, destino, reanudar=False, timeout=timeout)
        r.raise_for_status()
        if r.status_code != 206:
            offset = 0
        total = int(r.headers.get("content-length") or 0) + offset
        hechos = offset
        with open(part, "ab" if offset else "wb") as f:
            for trozo in r.iter_content(1024 * 1024):
                f.write(trozo)
                hechos += len(trozo)
                if total:
                    print(f"\r  {os.path.basename(destino)}: "
                          f"{hechos / 2**20:.1f}/{total / 2**20:.1f} MB "
                          f"({100 * hechos / total:.1f}%)", end="", flush=True)
        if total:
            print()
    os.replace(part, destino)


def leer_lista(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _elegir(nombres, cuota, destino, semilla):
    ya = set(os.listdir(destino))
    disponibles = [n for n in nombres if os.path.basename(n) not in ya]
    rng = random.Random(semilla)
    return rng.sample(disponibles, min(cuota, len(disponibles)))


def extraer_remotezip(url, test, cuota, destino, semilla):
    """Extrae solo los miembros necesarios usando HTTP Range (remotezip)."""
    from remotezip import RemoteZip
    with RemoteZip(url) as z:
        nombres = [n for n in z.namelist()
                   if n.startswith("images/") and n.lower().endswith(".png")
                   and os.path.basename(n) in test]
        elegidos = _elegir(nombres, cuota, destino, semilla)
        for n in elegidos:
            dst = os.path.join(destino, os.path.basename(n))
            with open(dst + ".tmp", "wb") as f:
                f.write(z.read(n))
            os.replace(dst + ".tmp", dst)
    return len(elegidos)


def extraer_zip_local(zip_path, test, cuota, destino, semilla):
    with zipfile.ZipFile(zip_path) as z:
        nombres = [n for n in z.namelist()
                   if n.startswith("images/") and n.lower().endswith(".png")
                   and os.path.basename(n) in test]
        elegidos = _elegir(nombres, cuota, destino, semilla)
        for n in elegidos:
            dst = os.path.join(destino, os.path.basename(n))
            with z.open(n) as src, open(dst + ".tmp", "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(dst + ".tmp", dst)
    return len(elegidos)


def preparar_nih(destino, zips, n_total, semilla):
    nih = os.path.join(destino, "NIH")
    imgs = os.path.join(nih, "images")
    tmp = os.path.join(destino, "_tmp")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    test_path = os.path.join(nih, "test_list.txt")
    if not os.path.exists(test_path):
        print("Descargando test_list.txt...")
        descargar(HF_NIH + "/test_list.txt", test_path)
    train_path = os.path.join(nih, "train_val_list.txt")
    if not os.path.exists(train_path):
        print("Descargando train_val_list.txt...")
        descargar(HF_NIH + "/train_val_list.txt", train_path)
    test = set(leer_lista(test_path))
    print(f"test_list.txt: {len(test)} nombres")

    ya = len(os.listdir(imgs))
    if ya >= n_total:
        print(f"NIH: ya hay {ya} PNG (>= {n_total}); no se descarga nada.")
        return {"ya_existentes": ya}
    faltan_total = n_total - ya
    cuota = [faltan_total // len(zips)] * len(zips)
    for i in range(faltan_total % len(zips)):
        cuota[i] += 1

    detalle = {}
    for z, c in zip(zips, cuota):
        nombre = f"images_{z}.zip"
        url = f"{HF_NIH}/images/{nombre}"
        print(f"\n{nombre}: buscando {c} imágenes de test...")
        try:
            hechos = extraer_remotezip(url, test, c, imgs, semilla + int(z))
            modo = "remotezip"
        except Exception as e:
            print(f"  remotezip falló ({type(e).__name__}: {e}); bajando zip completo...")
            local = os.path.join(tmp, nombre)
            if not os.path.exists(local):
                descargar(url, local)
            hechos = extraer_zip_local(local, test, c, imgs, semilla + int(z))
            if os.path.exists(local):
                os.remove(local)
            modo = "zip completo"
        detalle[nombre] = {"modo": modo, "extraidas": hechos}
        print(f"  {hechos} extraídas ({modo})")

    # Si alguna cuota no se pudo cubrir, completar con los zips que tengan sobrantes.
    faltan = n_total - len(os.listdir(imgs))
    if faltan > 0:
        print(f"\nFaltan {faltan}; completando desde los mismos zips...")
        for z in zips:
            if faltan <= 0:
                break
            nombre = f"images_{z}.zip"
            url = f"{HF_NIH}/images/{nombre}"
            try:
                hechos = extraer_remotezip(url, test, faltan, imgs, semilla + 1000 + int(z))
            except Exception:
                continue
            detalle[nombre]["extraidas"] += hechos
            faltan -= hechos

    print(f"\nNIH: {len(os.listdir(imgs))} PNG en {imgs}")
    return detalle


def preparar_chexpert(destino):
    chex = os.path.join(destino, "CheXpert-v1.0-small")
    valid = os.path.join(chex, "valid")
    tmp = os.path.join(destino, "_tmp")
    os.makedirs(valid, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    parquet = os.path.join(tmp, "chex_valid.parquet")
    if not os.path.exists(parquet):
        print("\nDescargando validación de CheXpert (parquet)...")
        descargar(HF_CHEX, parquet)

    df = pd.read_parquet(parquet)
    col_img = "image" if "image" in df.columns else next(
        (c for c in df.columns if "image" in c.lower()), None)
    if col_img is None or "Path" not in df.columns:
        raise SystemExit(f"Parquet inesperado. Columnas: {list(df.columns)}")
    print(f"CheXpert valid: {len(df)} filas, columnas: {list(df.columns)}")

    # Pasar los índices de clase de HF a los valores que espera CheX_Dataset.
    df = df.copy()
    for col, mapa in [("Sex", MAPA_SEXO), ("Frontal/Lateral", MAPA_VISTA),
                      ("AP/PA", MAPA_APPA)]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(mapa)
    for col in ETIQUETAS_CHEX:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(MAPA_ETIQUETA)

    for _, fila in df.iterrows():
        img = fila[col_img]
        datos = img.get("bytes") if isinstance(img, dict) else img
        if datos is None:
            raise SystemExit(f"Fila sin bytes de imagen: {fila['Path']}")
        rel = str(fila["Path"]).replace("CheXpert-v1.0-small/", "").replace("CheXpert-v1.0/", "")
        dst = os.path.join(chex, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(datos)

    df.drop(columns=[col_img]).to_csv(os.path.join(chex, "valid.csv"), index=False)
    n = sum(len(fs) for _, _, fs in os.walk(valid))
    print(f"CheXpert: {n} imágenes en {valid}")
    os.remove(parquet)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default=r"C:\datos")
    ap.add_argument("--n-nih", type=int, default=234)
    ap.add_argument("--zips", default="001,006,012")
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    zips = [z.strip().zfill(3) for z in a.zips.split(",") if z.strip()]
    os.makedirs(a.destino, exist_ok=True)
    t0 = time.time()
    detalle = preparar_nih(a.destino, zips, a.n_nih, a.semilla)
    n_chex = preparar_chexpert(a.destino)

    manifest = {
        "destino": a.destino,
        "nih": {
            "zips": zips,
            "n_solicitadas": a.n_nih,
            "n_disponibles": len(os.listdir(os.path.join(a.destino, "NIH", "images"))),
            "detalle": detalle,
        },
        "chexpert": {"n_imagenes": n_chex},
        "semilla": a.semilla,
        "fuentes": {"nih": HF_NIH, "chexpert": HF_CHEX},
        "segundos": round(time.time() - t0, 1),
    }
    ruta = os.path.join(a.destino, "manifest.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nListo en {manifest['segundos']} s. Manifiesto: {ruta}")


if __name__ == "__main__":
    main()
