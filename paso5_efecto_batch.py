#!/usr/bin/env python
"""PASO 5: comprobar cuantitativamente el efecto batch entre hospitales.

Extrae los embeddings de 2048 dims (penúltima capa de ResNet-50) de las mismas
imágenes procesadas por el modelo y comprueba si un clasificador puede distinguir
el hospital de origen. Es la prueba cuantitativa del efecto batch; el t-SNE es
solo la visualización.

Comparaciones:
  * hospital   : NIH test a 320 px  vs  CheXpert small valid a 320 px
                 (resolución emparejada para no confundir hospital con resolución)
  * resolución : NIH test a 512 px  vs  NIH test a 320 px
                 (control: mismo hospital, distinta resolución)

Métricas por comparación:
  * AUC y accuracy de un clasificador de dominio (regresión logística, 5-fold
    estratificado). AUC = 0.5 -> los dominios no se distinguen.
  * p-valor por permutación de las etiquetas de dominio.
  * silhouette de las etiquetas de dominio en el espacio PCA (50 dims).

Salidas:
    resultados/paso5/efecto_batch.json
    resultados/figuras/15_tsne_efecto_batch.png
    resultados/figuras/16_tsne_resolucion.png

Uso (desde la raíz del repo):
    python paso5_efecto_batch.py --nih "C:\\datos\\NIH" --chex "C:\\datos\\CheXpert-v1.0-small"
Prueba rápida:
    ... --n 50 --n-perm 100 --workers 0

Si --nih no contiene test_list.txt se usa train_val_list.txt. Si solo hay un
subconjunto de las imágenes descargadas, se usan las que existan en disco.
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comun_proyecto as cp  # noqa: E402


def _buscar_lista(nih):
    """Devuelve (ruta, nombre) de test_list.txt o, si no existe, train_val_list.txt."""
    for nombre in ("test_list.txt", "train_val_list.txt"):
        encontrados = glob.glob(os.path.join(nih, "**", nombre), recursive=True)
        if encontrados:
            return encontrados[0], nombre
    return None, None


def cargar_datos(a):
    """NIH test a 320 y 512 (mismas imágenes) y CheXpert valid a 320, balanceados.

    Acepta un subconjunto de NIH: se queda solo con las imágenes de la lista que
    existan en disco, para que funcione con descargas parciales.
    """
    ruta_lista, nombre_lista = _buscar_lista(a.nih)
    if ruta_lista is None:
        sys.exit("No se encontró test_list.txt ni train_val_list.txt dentro de --nih (ver paso 0).")
    if nombre_lista != "test_list.txt":
        print(f"Aviso: no hay test_list.txt; se usa {nombre_lista}.")
    lista = cp.leer_lista(ruta_lista)

    dir_orig = os.path.join(a.nih, "images")
    dir320 = os.path.join(a.nih, "images_320")
    if not os.path.isdir(dir320):
        print("Aviso: no existe images_320 (paso 1); se reducirá al vuelo, más lento.")
        dir320 = dir_orig

    presentes = set(os.listdir(dir_orig))
    if dir320 != dir_orig:
        presentes &= set(os.listdir(dir320))
    lista = {n for n in lista if n in presentes}
    if len(lista) < 10:
        sys.exit(f"Muy pocas imágenes de {nombre_lista} presentes en {dir_orig} ({len(lista)}).")
    print(f"{nombre_lista}: {len(lista)} imágenes presentes en disco.")

    nih320_full = cp.dataset_nih(dir320, lista, res=320)
    nih512_full = cp.dataset_nih(dir_orig, lista, res=512)
    chex_full = cp.dataset_chexpert(a.chex, "valid.csv", res=320)

    n = min(len(nih320_full), len(chex_full)) if a.n <= 0 else min(a.n, len(nih320_full), len(chex_full))
    if n < 10:
        sys.exit(f"Muy pocas imágenes por dominio ({n}); no se puede analizar.")

    nih320 = cp.limitar(nih320_full, n, seed=a.semilla)
    nih512 = cp.limitar(nih512_full, n, seed=a.semilla)
    chex320 = cp.limitar(chex_full, n, seed=a.semilla)

    assert list(nih320.csv["Image Index"]) == list(nih512.csv["Image Index"]), \
        "NIH-320 y NIH-512 deben usar exactamente las mismas imágenes."
    return nih320, nih512, chex320, n


def prueba_dominio(X, y, semilla, n_perm, n_pca=50):
    """¿Se puede predecir el dominio (hospital/resolución) desde el embedding?

    Estandariza y reduce con PCA antes de clasificar: con ~500 muestras y 2048
    dimensiones, la regresión logística sobre las dimensiones crudas se ajusta
    al ruido y además es mucho más lenta. El PCA no usa las etiquetas, así que
    se ajusta una sola vez y sirve también para el test de permutación.
    """
    n_comp = int(min(n_pca, X.shape[0] - 1, X.shape[1]))
    Xp = PCA(n_components=n_comp, random_state=semilla).fit_transform(
        StandardScaler().fit_transform(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
    clf = LogisticRegression(max_iter=2000, solver="liblinear",
                             class_weight="balanced", random_state=semilla)
    prob = cross_val_predict(clf, Xp, y, cv=cv, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, prob))
    acc = float(accuracy_score(y, (prob > 0.5).astype(int)))
    sil = float(silhouette_score(Xp, y))

    rng = np.random.default_rng(semilla)
    perm = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        yp = rng.permutation(y)
        p = cross_val_predict(clf, Xp, yp, cv=cv, method="predict_proba")[:, 1]
        perm[i] = roc_auc_score(yp, p)
    p_valor = float((1 + np.sum(perm >= auc)) / (1 + n_perm))

    return {"auc_dominio": auc, "accuracy": acc, "silhouette": sil,
            "auc_perm_media": float(perm.mean()), "auc_perm_p95": float(np.percentile(perm, 95)),
            "p_valor": p_valor}


def comparar(e_a, e_b, nombre_a, nombre_b, semilla, n_perm, n_pca):
    X = np.vstack([e_a, e_b]).astype(np.float32)
    y = np.concatenate([np.zeros(len(e_a), dtype=int), np.ones(len(e_b), dtype=int)])
    r = prueba_dominio(X, y, semilla, n_perm, n_pca)
    r.update({"dominios": [nombre_a, nombre_b], "n_a": int(len(e_a)), "n_b": int(len(e_b))})
    return r


def proyectar_tsne(X, semilla, perplexity):
    Xs = StandardScaler().fit_transform(X)
    n_comp = int(min(50, Xs.shape[0] - 1, Xs.shape[1]))
    Xp = PCA(n_components=n_comp, random_state=semilla).fit_transform(Xs)
    perp = float(min(perplexity, max(5, Xs.shape[0] - 1)))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=semilla,
                init="pca", learning_rate="auto")
    return tsne.fit_transform(Xp)


def graficar(Z, y, nombres, colores, titulo, ruta):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    for k, (nombre, color) in enumerate(zip(nombres, colores)):
        m = y == k
        ax.scatter(Z[m, 0], Z[m, 1], c=color, label=f"{nombre} (n={int(m.sum())})",
                   alpha=0.65, edgecolors="w", linewidth=0.4, s=40)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)


def interpretar(c):
    if c["p_valor"] < 0.05 and c["auc_dominio"] > 0.6:
        return "efecto batch confirmado"
    return "sin evidencia suficiente de efecto batch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nih", required=True, help="carpeta raíz de NIH (contiene images\\ y test_list.txt)")
    ap.add_argument("--chex", required=True, help="carpeta CheXpert-v1.0-small")
    ap.add_argument("--modelo", default="resnet50-res512-all")
    ap.add_argument("--n", type=int, default=234, help="imágenes por dominio (0 = máximo común)")
    ap.add_argument("--n-perm", type=int, default=1000, help="permutaciones del test de dominio")
    ap.add_argument("--pca", type=int, default=50, help="dimensiones PCA antes del clasificador")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out", default=os.path.join(cp.AQUI, "resultados", "paso5"))
    a = ap.parse_args()

    device = cp.bench.pick_device("auto")
    figuras = os.path.join(cp.AQUI, "resultados", "figuras")
    os.makedirs(figuras, exist_ok=True)

    nih320, nih512, chex320, n = cargar_datos(a)
    print(f"NIH-320: {len(nih320)} | NIH-512: {len(nih512)} | CheXpert-320: {len(chex320)} "
          f"| {n} por dominio | dispositivo: {device}", flush=True)

    modelo, _, _ = cp.cargar_modelo(a.modelo, device)
    print(f"Extrayendo embeddings de {a.modelo}...", flush=True)
    e_nih320 = cp.extraer_embeddings(modelo, nih320, device, a.bs, a.workers)
    e_nih512 = cp.extraer_embeddings(modelo, nih512, device, a.bs, a.workers)
    e_chex320 = cp.extraer_embeddings(modelo, chex320, device, a.bs, a.workers)
    print(f"Embeddings: {e_nih320.shape[1]} dims", flush=True)

    print("\n== Hospital (NIH-320 vs CheXpert-320)", flush=True)
    hospital = comparar(e_nih320, e_chex320, "NIH 320 px", "CheXpert 320 px", a.semilla, a.n_perm, a.pca)
    print(f"   AUC dominio {hospital['auc_dominio']:.3f} | accuracy {hospital['accuracy']:.3f} "
          f"| p-valor {hospital['p_valor']:.4f} | silhouette {hospital['silhouette']:.3f}", flush=True)

    print("\n== Resolución (NIH-512 vs NIH-320, control)", flush=True)
    resolucion = comparar(e_nih512, e_nih320, "NIH 512 px", "NIH 320 px", a.semilla, a.n_perm, a.pca)
    print(f"   AUC dominio {resolucion['auc_dominio']:.3f} | accuracy {resolucion['accuracy']:.3f} "
          f"| p-valor {resolucion['p_valor']:.4f} | silhouette {resolucion['silhouette']:.3f}", flush=True)

    print("\n== t-SNE", flush=True)
    Z_h = proyectar_tsne(np.vstack([e_nih320, e_chex320]), a.semilla, a.perplexity)
    graficar(Z_h, np.array([0] * len(e_nih320) + [1] * len(e_chex320)),
             ["NIH ChestX-ray14 (320 px)", "CheXpert (320 px)"], ["#2a78d6", "#eb6834"],
             "t-SNE de embeddings ResNet-50 — efecto hospital",
             os.path.join(figuras, "15_tsne_efecto_batch.png"))

    Z_r = proyectar_tsne(np.vstack([e_nih512, e_nih320]), a.semilla, a.perplexity)
    graficar(Z_r, np.array([0] * len(e_nih512) + [1] * len(e_nih320)),
             ["NIH 512 px", "NIH 320 px"], ["#1baf7a", "#eb6834"],
             "t-SNE de embeddings ResNet-50 — control de resolución",
             os.path.join(figuras, "16_tsne_resolucion.png"))

    R = {"modelo": a.modelo, "device": str(device), "env": cp.bench.env_info(device),
         "semilla": a.semilla, "n_perm": a.n_perm, "n_pca": a.pca, "n_por_dominio": n,
         "comparaciones": {"hospital": hospital, "resolucion": resolucion}}
    cp.guardar_json(R, os.path.join(a.out, "efecto_batch.json"))

    print("\nResumen:")
    print(f"  Hospital   : {interpretar(hospital)}")
    print(f"  Resolución : {interpretar(resolucion)}")
    print(f"\nListo: {os.path.join(a.out, 'efecto_batch.json')}")
    print(f"Figuras: {os.path.join(figuras, '15_tsne_efecto_batch.png')} y 16_tsne_resolucion.png")


if __name__ == "__main__":
    main()
