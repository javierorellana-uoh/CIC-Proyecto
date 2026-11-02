# Paso 2: línea base con modelos preentrenados

Dispositivo: cuda. Imágenes: nih_original=25596, nih_320=25596, chex_valid=202

## AUC promedio (7 patologías comunes)

| Modelo | NIH original | NIH 320 px | Brecha (orig - 320) | CheXpert valid (320) | Fidelidad: % decisiones que cambian | Spearman |
|---|--:|--:|--:|--:|--:|--:|
| resnet50-res512-all | 0.810 | 0.789 | +0.021 | 0.664 | 2.9 % | 0.914 |
| densenet121-res224-nih | 0.691 | 0.691 | -0.000 | 0.723 | 0.3 % | 1.000 |
| densenet121-res224-chex | 0.637 | 0.636 | +0.000 | 0.824 | 0.2 % | 1.000 |
| densenet121-res224-all | 0.732 | 0.731 | +0.001 | 0.801 | 1.3 % | 0.999 |

## NIH test, resolución original

| Modelo | Patología | AUC [IC95] | AUPRC | Prevalencia | Sens. | Espec. | Brier |
|---|---|---|--:|--:|--:|--:|--:|
| resnet50-res512-all | Atelectasis | 0.757 [0.74-0.77] | 0.321 | 0.128 | 0.05 | 0.99 | 0.100 |
| resnet50-res512-all | Cardiomegaly | 0.896 [0.88-0.91] | 0.375 | 0.042 | 0.11 | 1.00 | 0.033 |
| resnet50-res512-all | Consolidation | 0.752 [0.74-0.77] | 0.174 | 0.071 | 0.01 | 1.00 | 0.063 |
| resnet50-res512-all | Edema | 0.858 [0.84-0.87] | 0.194 | 0.036 | 0.02 | 1.00 | 0.032 |
| resnet50-res512-all | Effusion | 0.827 [0.82-0.84] | 0.512 | 0.182 | 0.30 | 0.96 | 0.116 |
| resnet50-res512-all | Pneumonia | 0.698 [0.68-0.72] | 0.047 | 0.022 | 0.14 | 0.95 | 0.055 |
| resnet50-res512-all | Pneumothorax | 0.884 [0.87-0.90] | 0.512 | 0.104 | 0.45 | 0.96 | 0.069 |
| densenet121-res224-nih | Atelectasis | 0.662 [0.65-0.68] | 0.201 | 0.128 | 0.90 | 0.29 | 0.110 |
| densenet121-res224-nih | Cardiomegaly | 0.808 [0.79-0.83] | 0.224 | 0.042 | 0.76 | 0.69 | 0.039 |
| densenet121-res224-nih | Consolidation | 0.664 [0.65-0.68] | 0.107 | 0.071 | 0.97 | 0.19 | 0.067 |
| densenet121-res224-nih | Edema | 0.735 [0.71-0.75] | 0.083 | 0.036 | 0.94 | 0.33 | 0.035 |
| densenet121-res224-nih | Effusion | 0.747 [0.73-0.76] | 0.395 | 0.182 | 0.87 | 0.45 | 0.141 |
| densenet121-res224-nih | Pneumonia | 0.621 [0.59-0.65] | 0.032 | 0.022 | 0.83 | 0.33 | 0.022 |
| densenet121-res224-nih | Pneumothorax | 0.600 [0.57-0.63] | 0.139 | 0.104 | 0.84 | 0.30 | 0.098 |
| densenet121-res224-chex | Atelectasis | 0.566 [0.55-0.58] | 0.142 | 0.128 | 0.90 | 0.21 | 0.373 |
| densenet121-res224-chex | Cardiomegaly | 0.623 [0.59-0.65] | 0.062 | 0.042 | 0.82 | 0.32 | 0.336 |
| densenet121-res224-chex | Consolidation | 0.674 [0.66-0.69] | 0.116 | 0.071 | 0.93 | 0.27 | 0.208 |
| densenet121-res224-chex | Edema | 0.697 [0.67-0.72] | 0.066 | 0.036 | 0.97 | 0.18 | 0.305 |
| densenet121-res224-chex | Effusion | 0.672 [0.66-0.69] | 0.266 | 0.182 | 0.90 | 0.32 | 0.337 |
| densenet121-res224-chex | Pneumonia | 0.585 [0.56-0.61] | 0.027 | 0.022 | 0.77 | 0.35 | 0.101 |
| densenet121-res224-chex | Pneumothorax | 0.641 [0.62-0.66] | 0.181 | 0.104 | 0.47 | 0.71 | 0.095 |
| densenet121-res224-all | Atelectasis | 0.717 [0.70-0.73] | 0.259 | 0.128 | 0.80 | 0.50 | 0.105 |
| densenet121-res224-all | Cardiomegaly | 0.846 [0.82-0.87] | 0.276 | 0.042 | 0.64 | 0.87 | 0.035 |
| densenet121-res224-all | Consolidation | 0.695 [0.68-0.71] | 0.120 | 0.071 | 0.90 | 0.37 | 0.066 |
| densenet121-res224-all | Edema | 0.774 [0.75-0.79] | 0.101 | 0.036 | 0.60 | 0.78 | 0.034 |
| densenet121-res224-all | Effusion | 0.780 [0.77-0.79] | 0.421 | 0.182 | 0.84 | 0.57 | 0.128 |
| densenet121-res224-all | Pneumonia | 0.652 [0.62-0.68] | 0.040 | 0.022 | 0.49 | 0.72 | 0.065 |
| densenet121-res224-all | Pneumothorax | 0.659 [0.63-0.68] | 0.158 | 0.104 | 0.89 | 0.36 | 0.098 |

## NIH test reducido a 320 px

| Modelo | Patología | AUC [IC95] | AUPRC | Prevalencia | Sens. | Espec. | Brier |
|---|---|---|--:|--:|--:|--:|--:|
| resnet50-res512-all | Atelectasis | 0.746 [0.73-0.76] | 0.306 | 0.128 | 0.04 | 1.00 | 0.101 |
| resnet50-res512-all | Cardiomegaly | 0.888 [0.87-0.90] | 0.353 | 0.042 | 0.04 | 1.00 | 0.035 |
| resnet50-res512-all | Consolidation | 0.739 [0.72-0.75] | 0.165 | 0.071 | 0.00 | 1.00 | 0.063 |
| resnet50-res512-all | Edema | 0.836 [0.82-0.85] | 0.174 | 0.036 | 0.00 | 1.00 | 0.032 |
| resnet50-res512-all | Effusion | 0.813 [0.80-0.82] | 0.493 | 0.182 | 0.12 | 0.99 | 0.121 |
| resnet50-res512-all | Pneumonia | 0.677 [0.65-0.70] | 0.046 | 0.022 | 0.00 | 1.00 | 0.025 |
| resnet50-res512-all | Pneumothorax | 0.826 [0.81-0.84] | 0.400 | 0.104 | 0.04 | 1.00 | 0.083 |
| densenet121-res224-nih | Atelectasis | 0.662 [0.65-0.68] | 0.201 | 0.128 | 0.90 | 0.29 | 0.110 |
| densenet121-res224-nih | Cardiomegaly | 0.808 [0.79-0.83] | 0.224 | 0.042 | 0.76 | 0.69 | 0.039 |
| densenet121-res224-nih | Consolidation | 0.663 [0.65-0.68] | 0.107 | 0.071 | 0.97 | 0.19 | 0.067 |
| densenet121-res224-nih | Edema | 0.735 [0.71-0.75] | 0.083 | 0.036 | 0.94 | 0.34 | 0.035 |
| densenet121-res224-nih | Effusion | 0.746 [0.73-0.76] | 0.395 | 0.182 | 0.87 | 0.45 | 0.141 |
| densenet121-res224-nih | Pneumonia | 0.622 [0.59-0.65] | 0.032 | 0.022 | 0.83 | 0.33 | 0.022 |
| densenet121-res224-nih | Pneumothorax | 0.599 [0.57-0.63] | 0.139 | 0.104 | 0.84 | 0.30 | 0.098 |
| densenet121-res224-chex | Atelectasis | 0.565 [0.55-0.58] | 0.142 | 0.128 | 0.90 | 0.21 | 0.374 |
| densenet121-res224-chex | Cardiomegaly | 0.622 [0.59-0.65] | 0.062 | 0.042 | 0.82 | 0.32 | 0.337 |
| densenet121-res224-chex | Consolidation | 0.674 [0.66-0.69] | 0.116 | 0.071 | 0.92 | 0.27 | 0.208 |
| densenet121-res224-chex | Edema | 0.697 [0.67-0.72] | 0.066 | 0.036 | 0.97 | 0.18 | 0.304 |
| densenet121-res224-chex | Effusion | 0.672 [0.66-0.69] | 0.265 | 0.182 | 0.90 | 0.32 | 0.337 |
| densenet121-res224-chex | Pneumonia | 0.585 [0.56-0.61] | 0.027 | 0.022 | 0.77 | 0.35 | 0.101 |
| densenet121-res224-chex | Pneumothorax | 0.640 [0.61-0.66] | 0.180 | 0.104 | 0.47 | 0.71 | 0.095 |
| densenet121-res224-all | Atelectasis | 0.716 [0.70-0.73] | 0.258 | 0.128 | 0.80 | 0.50 | 0.105 |
| densenet121-res224-all | Cardiomegaly | 0.845 [0.82-0.86] | 0.274 | 0.042 | 0.64 | 0.87 | 0.035 |
| densenet121-res224-all | Consolidation | 0.695 [0.68-0.71] | 0.120 | 0.071 | 0.90 | 0.37 | 0.066 |
| densenet121-res224-all | Edema | 0.774 [0.75-0.79] | 0.101 | 0.036 | 0.56 | 0.79 | 0.034 |
| densenet121-res224-all | Effusion | 0.780 [0.77-0.79] | 0.420 | 0.182 | 0.85 | 0.56 | 0.128 |
| densenet121-res224-all | Pneumonia | 0.649 [0.62-0.68] | 0.040 | 0.022 | 0.44 | 0.75 | 0.057 |
| densenet121-res224-all | Pneumothorax | 0.656 [0.63-0.68] | 0.156 | 0.104 | 0.89 | 0.36 | 0.098 |

## CheXpert small valid (externo)

| Modelo | Patología | AUC [IC95] | AUPRC | Prevalencia | Sens. | Espec. | Brier |
|---|---|---|--:|--:|--:|--:|--:|
| resnet50-res512-all | Atelectasis | 0.614 [0.54-0.69] | 0.498 | 0.371 | 0.01 | 1.00 | 0.324 |
| resnet50-res512-all | Cardiomegaly | 0.705 [0.63-0.78] | 0.592 | 0.327 | 0.00 | 1.00 | 0.311 |
| resnet50-res512-all | Consolidation | 0.763 [0.67-0.85] | 0.360 | 0.158 | 0.06 | 0.99 | 0.137 |
| resnet50-res512-all | Edema | 0.638 [0.53-0.74] | 0.456 | 0.208 | 0.00 | 1.00 | 0.206 |
| resnet50-res512-all | Effusion | 0.717 [0.64-0.80] | 0.568 | 0.317 | 0.06 | 1.00 | 0.280 |
| resnet50-res512-all | Pneumonia | 0.572 [0.37-0.83] | 0.123 | 0.040 | 0.12 | 0.99 | 0.043 |
| resnet50-res512-all | Pneumothorax | 0.642 [0.45-0.87] | 0.120 | 0.035 | 0.00 | 1.00 | 0.034 |
| densenet121-res224-nih | Atelectasis | 0.716 [0.64-0.78] | 0.522 | 0.371 | 0.93 | 0.42 | 0.288 |
| densenet121-res224-nih | Cardiomegaly | 0.725 [0.65-0.80] | 0.579 | 0.327 | 0.33 | 0.90 | 0.311 |
| densenet121-res224-nih | Consolidation | 0.780 [0.69-0.87] | 0.387 | 0.158 | 0.97 | 0.27 | 0.143 |
| densenet121-res224-nih | Edema | 0.786 [0.70-0.85] | 0.473 | 0.208 | 0.62 | 0.77 | 0.204 |
| densenet121-res224-nih | Effusion | 0.777 [0.71-0.84] | 0.630 | 0.317 | 0.86 | 0.54 | 0.204 |
| densenet121-res224-nih | Pneumonia | 0.660 [0.46-0.82] | 0.124 | 0.040 | 1.00 | 0.16 | 0.037 |
| densenet121-res224-nih | Pneumothorax | 0.620 [0.38-0.84] | 0.187 | 0.035 | 0.86 | 0.19 | 0.034 |
| densenet121-res224-chex | Atelectasis | 0.801 [0.74-0.86] | 0.646 | 0.371 | 0.99 | 0.35 | 0.199 |
| densenet121-res224-chex | Cardiomegaly | 0.792 [0.72-0.85] | 0.657 | 0.327 | 0.89 | 0.52 | 0.189 |
| densenet121-res224-chex | Consolidation | 0.866 [0.81-0.91] | 0.529 | 0.158 | 1.00 | 0.41 | 0.121 |
| densenet121-res224-chex | Edema | 0.826 [0.74-0.89] | 0.585 | 0.208 | 0.98 | 0.33 | 0.186 |
| densenet121-res224-chex | Effusion | 0.834 [0.77-0.89] | 0.668 | 0.317 | 0.94 | 0.49 | 0.195 |
| densenet121-res224-chex | Pneumonia | 0.926 [0.84-0.98] | 0.354 | 0.040 | 1.00 | 0.42 | 0.075 |
| densenet121-res224-chex | Pneumothorax | 0.722 [0.53-0.89] | 0.076 | 0.035 | 0.57 | 0.71 | 0.035 |
| densenet121-res224-all | Atelectasis | 0.806 [0.74-0.87] | 0.657 | 0.371 | 1.00 | 0.20 | 0.197 |
| densenet121-res224-all | Cardiomegaly | 0.799 [0.73-0.86] | 0.649 | 0.327 | 0.97 | 0.29 | 0.170 |
| densenet121-res224-all | Consolidation | 0.858 [0.81-0.91] | 0.462 | 0.158 | 1.00 | 0.19 | 0.144 |
| densenet121-res224-all | Edema | 0.863 [0.79-0.92] | 0.713 | 0.208 | 0.98 | 0.25 | 0.133 |
| densenet121-res224-all | Effusion | 0.859 [0.80-0.91] | 0.733 | 0.317 | 0.98 | 0.28 | 0.190 |
| densenet121-res224-all | Pneumonia | 0.756 [0.61-0.89] | 0.123 | 0.040 | 1.00 | 0.15 | 0.098 |
| densenet121-res224-all | Pneumothorax | 0.664 [0.44-0.92] | 0.088 | 0.035 | 1.00 | 0.06 | 0.040 |

## Velocidad (imágenes por segundo)

| Modelo | NIH original | NIH 320 | CheXpert |
|---|--:|--:|--:|
| resnet50-res512-all | 278 | 464 | 19 |
| densenet121-res224-nih | 284 | 775 | 20 |
| densenet121-res224-chex | 281 | 783 | 20 |
| densenet121-res224-all | 277 | 778 | 20 |