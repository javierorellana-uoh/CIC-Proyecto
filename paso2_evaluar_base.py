#!/usr/bin/env python
"""PASO 2: línea base con los modelos ya entrenados de TorchXRayVision (sin entrenar nada).

Para cada modelo mide, en las 7 patologías comunes:
  * NIH test en resolución original (1024 -> 512)   = "original"
  * NIH test reducido a 320 px                        = "small"
  * CheXpert small valid (234 imágenes, radiólogos)   = hospital externo, solo existe a ~320 px
Métricas: AUC con IC95 % (bootstrap por paciente), AUPRC, sensibilidad,
especificidad, Brier; la BRECHA (AUC original - AUC small) y la FIDELIDAD
(cuánto cambian las respuestas entre original y small para las mismas imágenes).

Uso (desde la raíz del repo):
    python proyecto\\paso2_evaluar_base.py --nih C:\\datos\\NIH --chex C:\\datos\\CheXpert-v1.0-small
Prueba rápida:
    python proyecto\\paso2_evaluar_base.py --nih ... --chex ... --max-imgs 500 --n-boot 100
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun_proyecto as cp  # noqa: E402

MODELOS = ["resnet50-res512-all", "densenet121-res224-nih",
           "densenet121-res224-chex", "densenet121-res224-all"]


def reporte_md(r):
    L = ["# Paso 2: línea base con modelos preentrenados\n",
         f"Dispositivo: {r['device']}. Imágenes: " +
         ", ".join(f"{k}={v}" for k, v in r["n_imagenes"].items()) + "\n",
         "## AUC promedio (7 patologías comunes)\n",
         "| Modelo | NIH original | NIH 320 px | Brecha (orig - 320) | CheXpert valid (320) | Fidelidad: % decisiones que cambian | Spearman |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    for m, v in r["modelos"].items():
        g = lambda c: v[c]["metricas"].get("PROMEDIO", {}).get("auc", float("nan"))
        f = v.get("fidelidad_nih", {})
        L.append(f"| {m} | {g('nih_original'):.3f} | {g('nih_320'):.3f} | {g('nih_original') - g('nih_320'):+.3f} | "
                 f"{g('chex_valid'):.3f} | {f.get('pct_decisiones_cambian', float('nan')):.1f} % | {f.get('spearman_medio', float('nan')):.3f} |")
    for cond, titulo in [("nih_original", "NIH test, resolución original"),
                         ("nih_320", "NIH test reducido a 320 px"),
                         ("chex_valid", "CheXpert small valid (externo)")]:
        L += [f"\n## {titulo}\n",
              "| Modelo | Patología | AUC [IC95] | AUPRC | Prevalencia | Sens. | Espec. | Brier |",
              "|---|---|---|--:|--:|--:|--:|--:|"]
        for m, v in r["modelos"].items():
            for p, x in v[cond]["metricas"].items():
                if p == "PROMEDIO":
                    continue
                L.append(f"| {m} | {p} | {x['auc']:.3f} [{x['auc_ic95'][0]:.2f}-{x['auc_ic95'][1]:.2f}] | "
                         f"{x['auprc']:.3f} | {x['prevalencia']:.3f} | {x['sensibilidad']:.2f} | "
                         f"{x['especificidad']:.2f} | {x['brier']:.3f} |")
    L += ["\n## Velocidad (imágenes por segundo)\n", "| Modelo | NIH original | NIH 320 | CheXpert |", "|---|--:|--:|--:|"]
    for m, v in r["modelos"].items():
        L.append(f"| {m} | {v['nih_original']['img_s']:.0f} | {v['nih_320']['img_s']:.0f} | {v['chex_valid']['img_s']:.0f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", required=True, help="carpeta raíz de NIH (contiene images\\ y test_list.txt)")
    ap.add_argument("--chex", required=True, help="carpeta CheXpert-v1.0-small")
    ap.add_argument("--modelos", default=",".join(MODELOS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--max-imgs", type=int, default=0, help="limitar NIH test (prueba rápida)")
    ap.add_argument("--out", default=os.path.join(cp.AQUI, "resultados", "paso2"))
    ap.add_argument("--agregar", action="store_true",
                    help="agregar estos modelos a resultados.json existente en vez de reemplazarlo")
    a = ap.parse_args()
    device = cp.bench.pick_device(a.device)

    test_list = glob.glob(os.path.join(a.nih, "**", "test_list.txt"), recursive=True)
    if not test_list:
        sys.exit("No se encontró test_list.txt dentro de --nih (ver paso 0).")
    lista = cp.leer_lista(test_list[0])
    dir320 = os.path.join(a.nih, "images_320")
    if not os.path.isdir(dir320):
        print("Aviso: no existe images_320 (paso 1); se reducirá al vuelo, más lento.")
        dir320 = os.path.join(a.nih, "images")

    datos = {
        "nih_original": cp.limitar(cp.dataset_nih(os.path.join(a.nih, "images"), lista, res=512), a.max_imgs),
        "nih_320": cp.limitar(cp.dataset_nih(dir320, lista, res=320), a.max_imgs),
        "chex_valid": cp.dataset_chexpert(a.chex, "valid.csv", res=320),
    }
    assert list(datos["nih_original"].csv["Image Index"]) == list(datos["nih_320"].csv["Image Index"])
    print({k: len(v) for k, v in datos.items()}, "| dispositivo:", device)

    r = {"device": str(device), "env": cp.bench.env_info(device),
         "n_imagenes": {k: len(v) for k, v in datos.items()}, "modelos": {}}
    previo = os.path.join(a.out, "resultados.json")
    if a.agregar and os.path.exists(previo):
        with open(previo, encoding="utf-8") as f:
            anterior = json.load(f)
        if anterior.get("n_imagenes") != r["n_imagenes"]:
            sys.exit(f"Los resultados previos usan otras imágenes ({anterior.get('n_imagenes')}); "
                     "corran con los mismos argumentos (--max-imgs) o sin --agregar.")
        r["modelos"] = anterior["modelos"]
    for nombre in a.modelos.split(","):
        clave = cp.nombre_modelo(nombre)
        print(f"\n== {clave}", flush=True)
        modelo, targets, umbrales = cp.cargar_modelo(nombre, device)
        rm, decis = {}, {}
        for cond, d in datos.items():
            probs, seg = cp.predecir(modelo, d, device, a.bs, a.workers)
            decis[cond] = cp.decisiones(probs, umbrales)
            met = cp.metricas(d.labels, list(d.pathologies), probs, decis[cond], targets,
                              d.csv["patientid"].values, a.n_boot)
            rm[cond] = {"metricas": met, "img_s": len(d) / seg}
            print(f"   {cond:13s} AUC prom = {met.get('PROMEDIO', {}).get('auc', float('nan')):.3f}"
                  f"   ({len(d) / seg:.0f} img/s)", flush=True)
        rm["fidelidad_nih"] = cp.fidelidad(decis["nih_original"], decis["nih_320"], targets)
        print("   fidelidad original vs 320:", {k: round(v, 3) for k, v in rm["fidelidad_nih"].items()})
        r["modelos"][clave] = rm
        cp.guardar_json(r, os.path.join(a.out, "resultados.json"))
        del modelo

    with open(os.path.join(a.out, "REPORTE.md"), "w", encoding="utf-8") as f:
        f.write(reporte_md(r))
    print("\nListo:", os.path.join(a.out, "REPORTE.md"))


if __name__ == "__main__":
    main()
