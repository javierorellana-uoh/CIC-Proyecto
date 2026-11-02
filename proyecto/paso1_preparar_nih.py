#!/usr/bin/env python
"""PASO 1: crear la versión reducida de NIH (320 px por defecto).

Lee cada PNG de 1024x1024 de NIH\\images y guarda una copia reducida en
NIH\\images_<res> con el mismo nombre. Se hace una sola vez; si se interrumpe,
al relanzarlo continúa donde quedó.

Cada archivo se escribe primero como temporal y solo se renombra si quedó
completo, así un error de escritura (p. ej. disco lleno) nunca deja una imagen
rota. Con --verificar se relee cada imagen ya creada y se rehacen las dañadas.

Uso:
    python proyecto\\paso1_preparar_nih.py --nih C:\\datos\\NIH --res 320
    python proyecto\\paso1_preparar_nih.py --nih C:\\datos\\NIH --res 320 --verificar
"""
import argparse
import glob
import os
import shutil
import sys
import time
from multiprocessing import Pool

import cv2

NUEVA, YA_EXISTIA, ERROR_LECTURA, ERROR_ESCRITURA, REPARADA = range(5)


def valida(path, res):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img is not None and img.shape == (res, res)


def reducir(args):
    src, dst, res, verificar = args
    reparada = False
    if os.path.exists(dst):
        if not verificar or valida(dst, res):
            return YA_EXISTIA, dst
        reparada = True
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ERROR_LECTURA, src
    h, w = img.shape
    s = min(h, w)                                   # recorte central (NIH ya es cuadrada)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)  # reducción sin aliasing
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ERROR_ESCRITURA, dst
    tmp = dst + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, dst)
    except OSError as e:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return ERROR_ESCRITURA, f"{dst}: {e}"
    return (REPARADA if reparada else NUEVA), dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", required=True, help="carpeta raíz de NIH (contiene images\\)")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--verificar", action="store_true",
                    help="releer las imágenes ya creadas y rehacer las dañadas")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    origen = os.path.join(a.nih, "images")
    destino = os.path.join(a.nih, f"images_{a.res}")
    os.makedirs(destino, exist_ok=True)
    for t in glob.glob(os.path.join(destino, "*.tmp")):   # restos de una ejecución anterior
        os.remove(t)

    libre_gb = shutil.disk_usage(destino).free / 2**30
    print(f"Espacio libre en el disco de destino: {libre_gb:.1f} GB")
    if libre_gb < 5:
        print("AVISO: queda poco espacio. La versión de 320 px ocupa varios GB; libere espacio antes de seguir.")

    archivos = sorted(glob.glob(os.path.join(origen, "*.png")))
    tareas = [(f, os.path.join(destino, os.path.basename(f)), a.res, a.verificar) for f in archivos]
    print(f"{len(archivos)} imágenes -> {destino} ({a.res}x{a.res}) con {a.procesos} procesos"
          + (" [verificando las existentes]" if a.verificar else ""))

    t0 = time.time()
    cuenta = [0] * 5
    errores = []
    with Pool(a.procesos) as pool:
        for i, (r, info) in enumerate(pool.imap_unordered(reducir, tareas, chunksize=64), 1):
            cuenta[r] += 1
            if r in (ERROR_LECTURA, ERROR_ESCRITURA):
                errores.append(info)
            if i % 5000 == 0 or i == len(tareas):
                print(f"  {i}/{len(tareas)}  ({time.time() - t0:.0f} s)  errores: {len(errores)}", flush=True)

    print(f"\nNuevas: {cuenta[NUEVA]} | ya existían y están bien: {cuenta[YA_EXISTIA]} | "
          f"reparadas: {cuenta[REPARADA]} | error al leer: {cuenta[ERROR_LECTURA]} | "
          f"error al escribir: {cuenta[ERROR_ESCRITURA]}")
    if errores:
        print("Primeros errores:")
        for e in errores[:10]:
            print("  ", e)
        print("Si son errores de escritura, lo más probable es que el disco esté lleno: libere espacio y "
              "vuelva a ejecutar con --verificar.")
        sys.exit(1)
    print("Listo, sin errores.")


if __name__ == "__main__":
    main()
