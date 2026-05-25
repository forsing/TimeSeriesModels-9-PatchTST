#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Polazni kod treba samo da se promeni da radi nad mojim CSV-om, a sintetička od demoa izbaciti. 

Razumeo. Pravilo za sve buduće modele:

polazni kod iz članka se direktno menja da radi nad tvojim loto CSV-om
sintetička demo data, neiskorišćeni delovi, sve što ne pripada polaznom zadatku se izbacuje
predviđa se sledeće loto kolo + back-test, snimanje u TXT
bez paralelnih "novih" klasa ispod polaznog, bez pitanja




Jasno. Ponavljam svojim rečima da budem siguran:

Polazni kod iz članka prvo mora da proradi kakav jeste (sintetički demo, ništa ne ruši).
Tek posle se polazni direktno menja da radi nad tvojim loto CSV-om — sintetička demo data se izbacuje, svi neiskorišćeni delovi se izbacuju.
Cilj v2 fajla: predviđa sledeće loto kolo + back-test, snimanje rezultata u TXT.
Bez paralelnih „novih" klasa ispod polaznog, bez postavljanja pitanja kad je sve jasno.
"""




"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

9. PatchTST: Channel-Independent Patching Transformer

pip install neuralforecast


Pravi se Loto 7/39 NeuralForecast format sa 39 serija (po jedna za svaki broj), back-test na poslednjih 100 kola i finalni model za sledeće kolo.

Za PatchTST ne pravim ručni PyTorch model: NeuralForecast + PatchTST obrazac. Loto se predstavlja kao 39 nezavisnih kanala/serija, što je baš poenta PatchTST channel-independent pristupa.

Radi ovako:
39 nezavisnih serija, po jedna za svaki broj 1..39.
Back-test: trenira bez poslednjih 100 kola, predviđa narednih 100 horizonata.
Final: trenira na svim kolima, predviđa sledeće kolo.
Snima u 9_PatchTST_loto_v2_predikcija.txt.

max_steps=500 — to je 500 batch koraka, ne epoha. 
Lightning prikazuje "Epoch N" gde N ide do otprilike max_steps / batch_per_epoch. 
"""


import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.losses.pytorch import MSE

# =========================
# Loto 7/39 adaptacija (loto7hh_4620_k41.csv) — demo izbačen
# =========================
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score

SEED = 39
random.seed(SEED)
np.random.seed(SEED)

CSV_PATH = "/Users/4c/Desktop/GHQ/KvantniRegresor/loto7hh_4620_k41.csv"
OUT_TXT = Path("/Users/4c/Desktop/GHQ/TimeSeriesModels/9_PatchTST_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
BACKTEST_N = 100
INPUT_SIZE = 336
PATCH_LEN = 24
STRIDE = 12
HIDDEN_SIZE = 128
N_HEADS = 16
MAX_STEPS = 200 # Epoch

T0 = time.time()
print()
print("START 9_PatchTST_loto_v2", datetime.today())
print()

raw = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(raw.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def make_nf_df(y_multi):
    dates = pd.date_range("2000-01-01", periods=y_multi.shape[0], freq="D")
    parts = []
    for n in range(N_MAX):
        parts.append(pd.DataFrame({
            "unique_id": f"n{n + 1:02d}",
            "ds": dates,
            "y": y_multi[:, n].astype(np.float32),
        }))
    return pd.concat(parts, ignore_index=True)


def make_patchtst(h):
    # PatchTST input: 336 kola -> 27 patch-eva (patch_len=24, stride=12)
    return PatchTST(
        h=h,
        input_size=INPUT_SIZE,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        hidden_size=HIDDEN_SIZE,
        n_heads=N_HEADS,
        dropout=0.2,
        loss=MSE(),
        scaler_type="standard",
        max_steps=MAX_STEPS,
        accelerator="cpu",  # MPS pravi probleme na Apple GPU
    )


def forecast_matrix(pred_df):
    pred_col = [c for c in pred_df.columns if c not in ("unique_id", "ds")][0]
    wide = pred_df.pivot(index="ds", columns="unique_id", values=pred_col)
    wide = wide[[f"n{i:02d}" for i in range(1, N_MAX + 1)]]
    return wide.to_numpy(dtype=float)


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
Y_back = Y_full[-BACKTEST_N:]

df_all = make_nf_df(Y_full)
df_train_back = make_nf_df(Y_full[:-BACKTEST_N])

print(f"PatchTST series: {N_MAX}, input_size={INPUT_SIZE}, h(backtest)={BACKTEST_N}, max_steps={MAX_STEPS}")
print()

# Back-test: trenira do poslednjih 100 kola, pa predviđa tih 100 horizonata za svaki broj.
back_model = make_patchtst(h=BACKTEST_N)
nf_back = NeuralForecast(models=[back_model], freq="D")
print("Treniranje PatchTST za back-test ...")
nf_back.fit(df=df_train_back)
back_pred = nf_back.predict()
scores_back = forecast_matrix(back_pred)
h_back = avg_hits(scores_back, Y_back)
auc_back = safe_auc(Y_back, scores_back)
lrap_back = safe_lrap(Y_back, scores_back)

# Finalni model: trenira na svim kolima i predviđa sledeće kolo (h=1).
final_model = make_patchtst(h=1)
nf_final = NeuralForecast(models=[final_model], freq="D")
print()
print("Treniranje PatchTST za sledeće kolo ...")
nf_final.fit(df=df_all)
next_pred = nf_final.predict()
next_scores = forecast_matrix(next_pred)[0]
pick_next = topk_from_scores(next_scores)

assert len(set(pick_next.tolist())) == K, "PatchTST nema 7 jedinstvenih brojeva"
assert pick_next.min() >= N_MIN and pick_next.max() <= N_MAX, "PatchTST van opsega"
assert list(pick_next) == sorted(pick_next.tolist()), "PatchTST nije sortiran"

print()
print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"PatchTST -> {pick_next.tolist()}  ({describe(pick_next)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<12} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'PatchTST':<12} {h_back:>8.3f} {100*h_back/K:>6.1f}% {auc_back:>7.3f} {lrap_back:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()

elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, max_steps={MAX_STEPS}) ---\n")
    f.write(f"PatchTST -> {pick_next.tolist()}  ({describe(pick_next)})\n")
    f.write(
        f"back-test: hits/7={h_back:.3f}, AUC={auc_back:.3f}, LRAP={lrap_back:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")


"""
START 9_PatchTST_loto_v2 2026-05-25 20:10:07.766840

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

PatchTST series: 39, input_size=336, h(backtest)=100, max_steps=200

Seed set to 1
Treniranje PatchTST za back-test ...
GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores

  | Name         | Type              | Params | Mode 
-----------------------------------------------------------
0 | loss         | MSE               | 0      | train
1 | padder_train | ConstantPad1d     | 0      | train
2 | scaler       | TemporalNorm      | 0      | train
3 | model        | PatchTST_backbone | 762 K  | train
-----------------------------------------------------------
762 K     Trainable params
3         Non-trainable params
762 K     Total params
3.051     Total estimated model params size (MB)
90        Modules in train mode
0         Modules in eval mode
Epoch 99: 100%|█| 2/2 [00:01<00:00,  1.09it/s, v_num=2, train_los`Trainer.fit` stopped: `max_steps=200` reached.                  
Epoch 99: 100%|█| 2/2 [00:01<00:00,  1.09it/s, v_num=2, train_los
GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores
Predicting DataLoader 0: 100%|████| 2/2 [00:00<00:00, 111.69it/s]
Seed set to 1

Treniranje PatchTST za sledeće kolo ...
GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores

  | Name         | Type              | Params | Mode 
-----------------------------------------------------------
0 | loss         | MSE               | 0      | train
1 | padder_train | ConstantPad1d     | 0      | train
2 | scaler       | TemporalNorm      | 0      | train
3 | model        | PatchTST_backbone | 407 K  | train
-----------------------------------------------------------
407 K     Trainable params
3         Non-trainable params
407 K     Total params
1.631     Total estimated model params size (MB)
90        Modules in train mode
0         Modules in eval mode
Epoch 99: 100%|█| 2/2 [00:01<00:00,  1.16it/s, v_num=4, train_los`Trainer.fit` stopped: `max_steps=200` reached.                  
Epoch 99: 100%|█| 2/2 [00:01<00:00,  1.16it/s, v_num=4, train_los
GPU available: True (mps), used: False
TPU available: False, using: 0 TPU cores
Predicting DataLoader 0: 100%|████| 2/2 [00:00<00:00, 127.19it/s]

Predikcija sledeće Loto 7/39 kombinacije:
PatchTST -> [6, 10, 13, 18, 19, 32, 33]  (suma=131, neparnih=3/7, niskih(<=19)=5/7, raspon=27)

Back-test (poslednjih 100 izvlačenja):
model          hits/7    hit%     AUC    LRAP
PatchTST        1.240   17.7%   0.496   0.244
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /9_PatchTST_loto_v2_predikcija.txt

STOP 2026-05-25 20:15:58.649022
Ukupno vreme: 0:05:50  (350.9 s)
"""
