"""Paso 4: eficiencia de los modelos (latencia, velocidad, memoria y cómputo).

Mide, con imágenes sintéticas (el contenido no cambia el tiempo), cuánto cuesta usar cada
configuración del ResNet-50:

  * original-512        : modelo de la librería con su imagen de 512 px (lo "estándar")
  * original-320        : el mismo modelo recibiendo 320 px (por dentro la agranda a 512)
  * nuevo-320           : nuestro ResNet-50 ajustado a 320 px (procesa 320 px de verdad)

Métricas
  latencia_ms           : tiempo para diagnosticar UNA radiografía (lote de 1); mediana y p95
  imgs_por_s            : cuántas radiografías por segundo con lotes de 32 (rendimiento)
  vram_mb_lote1/32      : memoria máxima de GPU usada al predecir (lote de 1 y de 32)
  gflops_por_img        : operaciones de cálculo por radiografía (miles de millones)
  parametros_m, MB      : tamaño del modelo

Uso (desde la raíz del repo):
    python proyecto\\paso4_eficiencia.py
    python proyecto\\paso4_eficiencia.py --cpu          (agrega latencia en CPU, más lento)
Resultado: proyecto\\resultados\\paso4\\eficiencia.json y EFICIENCIA.md
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import comun_proyecto as cp  # noqa: E402

NUEVO = os.path.join(AQUI, "modelos", "resnet50-res320-chex-s0")


def configuraciones():
    c = [("original-512", "resnet50-res512-all", 512),
         ("original-320", "resnet50-res512-all", 320)]
    if os.path.isfile(os.path.join(NUEVO, "config.json")):
        c.append(("nuevo-320", NUEVO, 320))
    else:
        print("Aviso: no se encontró", NUEVO, "- se mide solo el modelo original.")
    return c


def sincronizar(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def entrada(bs, res, device):
    return (torch.rand(bs, 1, res, res, device=device) * 2048 - 1024)   # rango de TorchXRayVision


@torch.inference_mode()
def cronometrar(modelo, x, device, amp, repeticiones, calentamiento):
    usar_amp = amp and str(device).startswith("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=usar_amp), cp.bench.quiet():
        for _ in range(calentamiento):                  # las primeras pasadas son lentas (preparación)
            modelo(x)
        sincronizar(device)
        tiempos = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            modelo(x)
            sincronizar(device)                         # esperar a que la GPU termine de verdad
            tiempos.append(time.perf_counter() - t0)
    return np.array(tiempos)


@torch.inference_mode()
def vram_pico(modelo, x, amp):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp), cp.bench.quiet():
        modelo(x)
    torch.cuda.synchronize()
    return {"pico_total_mb": torch.cuda.max_memory_allocated() / 2**20,
            "pico_sobre_pesos_mb": (torch.cuda.max_memory_allocated() - base) / 2**20}


def gflops(modelo, res):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    x = entrada(1, res, next(modelo.parameters()).device).float()
    with torch.inference_mode(), cp.bench.quiet(), FlopCounterMode(display=False) as fc:
        modelo(x)
    return fc.get_total_flops() / 2 / 1e9          # multiplicación+suma = 1 operación (MAC)


def medir(nombre, fuente, res, device, a):
    modelo, _, _ = cp.cargar_modelo(fuente, device)
    modelo = modelo.eval()
    p = sum(t.numel() for t in modelo.parameters())
    r = {"configuracion": nombre, "modelo": cp.nombre_modelo(fuente), "resolucion_entrada": res,
         "resolucion_interna": getattr(modelo, "input_resolution", res),
         "parametros_m": p / 1e6, "tamano_mb_fp32": p * 4 / 2**20}
    r["gflops_por_img"] = gflops(modelo, res)

    t1 = cronometrar(modelo, entrada(1, res, device), device, a.amp, a.rep, a.calentamiento)
    r["latencia_ms"] = {"mediana": float(np.median(t1) * 1e3), "p95": float(np.percentile(t1, 95) * 1e3)}
    tb = cronometrar(modelo, entrada(a.bs, res, device), device, a.amp, max(a.rep // 4, 5), 3)
    r["imgs_por_s"] = float(a.bs / np.median(tb))

    if str(device).startswith("cuda"):
        r["vram_mb_lote1"] = vram_pico(modelo, entrada(1, res, device), a.amp)
        r[f"vram_mb_lote{a.bs}"] = vram_pico(modelo, entrada(a.bs, res, device), a.amp)

    if a.cpu:
        m_cpu, _, _ = cp.cargar_modelo(fuente, "cpu")
        tc = cronometrar(m_cpu.eval(), entrada(1, res, "cpu"), "cpu", False, a.rep_cpu, 2)
        r["latencia_cpu_ms"] = {"mediana": float(np.median(tc) * 1e3), "p95": float(np.percentile(tc, 95) * 1e3),
                                "hilos": torch.get_num_threads()}
        del m_cpu
    del modelo
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return r


def reporte(res, entorno, a):
    gpu = "gpu" in entorno
    L = ["# Paso 4: eficiencia del ResNet-50", "",
         f"Dispositivo: {entorno.get('gpu') or entorno.get('device')} · precisión: "
         f"{'fp16 (mixta)' if a.amp and gpu else 'fp32'} · lote para velocidad: {a.bs} · imágenes sintéticas", ""]
    cab = "| Configuración | Entra | Procesa | GFLOPs/img | Latencia {} (ms) | Imágenes/s | VRAM lote 1 (MB) | VRAM lote {} (MB) |".format(
        "GPU" if gpu else "CPU", a.bs)
    if a.cpu:
        cab += " Latencia CPU (ms) |"
    L += [cab, "|" + "---|" * (cab.count("|") - 1)]
    for r in res:
        v1 = r.get("vram_mb_lote1", {}).get("pico_total_mb")
        vb = r.get(f"vram_mb_lote{a.bs}", {}).get("pico_total_mb")
        fila = (f"| {r['configuracion']} | {r['resolucion_entrada']} px | {r['resolucion_interna']} px | "
                f"{r['gflops_por_img']:.1f} | {r['latencia_ms']['mediana']:.1f} | {r['imgs_por_s']:.0f} | "
                f"{'' if v1 is None else f'{v1:.0f}'} | {'' if vb is None else f'{vb:.0f}'} |")
        if a.cpu:
            fila += f" {r['latencia_cpu_ms']['mediana']:.0f} |"
        L.append(fila)
    L += ["", f"Parámetros: {res[0]['parametros_m']:.1f} millones ({res[0]['tamano_mb_fp32']:.0f} MB en fp32) en todas las configuraciones.",
          "", "- **Latencia**: tiempo para una radiografía (mediana de varias repeticiones).",
          "- **Imágenes/s**: rendimiento con lotes; es lo que importa para procesar muchas.",
          "- **VRAM**: memoria máxima de GPU al predecir (incluye los pesos del modelo).",
          "- `original-320` recibe 320 px pero por dentro la agranda a 512: cuesta casi lo mismo que `original-512`.",
          "  El ahorro real solo aparece con un modelo que procesa 320 px de verdad (`nuevo-320`)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bs", type=int, default=32, help="lote para medir imágenes/s")
    ap.add_argument("--rep", type=int, default=100, help="repeticiones para la latencia en GPU")
    ap.add_argument("--calentamiento", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="medir también la latencia en CPU (lote de 1)")
    ap.add_argument("--rep-cpu", type=int, default=20)
    ap.add_argument("--fp32", action="store_true", help="sin precisión mixta (por defecto fp16 en GPU, como en la evaluación)")
    ap.add_argument("--out", default=os.path.join(AQUI, "resultados", "paso4"))
    a = ap.parse_args()
    a.amp = not a.fp32
    torch.manual_seed(0)
    device = cp.bench.pick_device("auto")
    entorno = cp.bench.env_info(device)
    print("Dispositivo:", entorno.get("gpu") or device)

    res = []
    for nombre, fuente, r in configuraciones():
        print(f"Midiendo {nombre} ...", flush=True)
        res.append(medir(nombre, fuente, r, device, a))
        x = res[-1]
        print(f"  latencia {x['latencia_ms']['mediana']:.1f} ms · {x['imgs_por_s']:.0f} imgs/s · "
              f"{x['gflops_por_img']:.1f} GFLOPs"
              + (f" · VRAM lote {a.bs}: {x[f'vram_mb_lote{a.bs}']['pico_total_mb']:.0f} MB" if f"vram_mb_lote{a.bs}" in x else ""))

    os.makedirs(a.out, exist_ok=True)
    cp.guardar_json({"device": str(device), "env": entorno, "amp_fp16": a.amp, "bs": a.bs,
                     "resultados": res}, os.path.join(a.out, "eficiencia.json"))
    with open(os.path.join(a.out, "EFICIENCIA.md"), "w", encoding="utf-8") as f:
        f.write(reporte(res, entorno, a))
    print("\nListo:", os.path.join(a.out, "EFICIENCIA.md"))


if __name__ == "__main__":
    main()
