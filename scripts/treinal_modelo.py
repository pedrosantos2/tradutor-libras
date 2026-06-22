import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from config import COORD_SIZE, FRAME_WINDOW, SIGNS

DATA_PATH = "DATA"
ACTIONS = np.array(SIGNS)
EXPECTED_SHAPE = (FRAME_WINDOW, COORD_SIZE)
LABEL_MAP = {label: num for num, label in enumerate(ACTIONS)}

# --- CARREGA DATASET ---
sequences, labels = [], []

print("--- ANALISANDO DATASET ---")
for action in ACTIONS:
    dir_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(dir_path):
        continue
    for file in os.listdir(dir_path):
        if not file.endswith(".npy"):
            continue
        res = np.load(os.path.join(dir_path, file))
        if res.shape == EXPECTED_SHAPE:
            sequences.append(res)
            labels.append(LABEL_MAP[action])
        else:
            print(f"⚠️  Ignorando shape inválido: {file} | {res.shape}")

if not sequences:
    print("❌ Nenhum dado válido encontrado. Verifique se o dataset está processado.")
    raise SystemExit(1)

X = np.array(sequences)
y = np.array(labels)  # inteiros para stratify

print(f"\n{len(sequences)} sequências carregadas de {len(np.unique(y))} classes")

# --- SPLIT 80 / 10 / 10 (train / val / test) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

n_classes = len(ACTIONS)
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat   = to_categorical(y_val,   num_classes=n_classes)
y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

print(f"Split — treino: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}")

# --- MODELO LSTM ---
model = Sequential([
    LSTM(64,  return_sequences=True, input_shape=(FRAME_WINDOW, COORD_SIZE)),
    LSTM(128, return_sequences=True),
    LSTM(64,  return_sequences=False),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(n_classes, activation="softmax"),
])
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
model.summary()

callbacks = [
    ModelCheckpoint(
        "modelo_libras.keras",
        monitor="val_categorical_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=40,
        restore_best_weights=True,
        verbose=1,
    ),
]

print(f"\n--- TREINANDO COM {len(X_train)} SEQUÊNCIAS ---")
model.fit(
    X_train, y_train_cat,
    epochs=300,
    batch_size=8,
    validation_data=(X_val, y_val_cat),
    callbacks=callbacks,
)

# --- AVALIAÇÃO NO TEST SET ---
print("\n--- MÉTRICAS NO TEST SET ---")
y_pred = model.predict(X_test, verbose=0)
y_pred_cls = np.argmax(y_pred, axis=1)

present_classes = sorted(np.unique(np.concatenate([y_test, y_pred_cls])))
target_names = [ACTIONS[i] for i in present_classes]

print(classification_report(y_test, y_pred_cls, labels=present_classes, target_names=target_names))

print("Matriz de Confusão (linhas=real, colunas=predito):")
cm = confusion_matrix(y_test, y_pred_cls, labels=present_classes)
header = "".join(f"{n:>14}" for n in target_names)
print(f"{'':>14}{header}")
for i, row in enumerate(cm):
    print(f"{target_names[i]:>14}", "".join(f"{v:>14}" for v in row))

# --- SALVA labels.json SIDECAR ---
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(list(ACTIONS), f, ensure_ascii=False)

print("\n✅ Modelo salvo: modelo_libras.keras")
print("✅ Labels salvas: labels.json")
