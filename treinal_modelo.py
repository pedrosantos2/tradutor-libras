import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# --- CONFIGURAÇÕES ---
DATA_PATH = 'DATA'
ACTIONS = np.array([
    "OI", "TCHAU", "EU", "NOME", "OBRIGADO", "SIM", "NAO",
    "POR_FAVOR", "DESCULPA", "BEM", "GOSTAR", "AJUDA",
    "ENTENDER", "NAO_ENTENDER", "REPETIR", "PRAZER", "AMIGO", "SURDO",
    "MELANCIA", "LARANJA"
])
EXPECTED_SHAPE = (30, 126) # 30 frames, 126 coordenadas (2 mãos)
LABEL_MAP = {label:num for num, label in enumerate(ACTIONS)}

sequences, labels = [], []

print("--- ANALISANDO DATASET ---")
for action in ACTIONS:
    dir_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(dir_path):
        continue
        
    files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
    
    for file in files:
        file_path = os.path.join(dir_path, file)
        res = np.load(file_path)
        
        # VERIFICAÇÃO DE SEGURANÇA
        if res.shape == EXPECTED_SHAPE:
            sequences.append(res)
            labels.append(LABEL_MAP[action])
        else:
            print(f"⚠️ Ignorando arquivo corrompido/antigo: {file} | Shape: {res.shape}")

# Se o erro persistir aqui, é porque o 'sequences' está vazio ou inconsistente
if len(sequences) == 0:
    print("❌ Erro: Nenhum dado válido encontrado. Verifique se gravou os sinais com 2 mãos.")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Divide os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=labels)

# --- MODELO LSTM ---
# --- MODELO LSTM BALANCEADO ---
model = Sequential([
    # As LSTMs vão focar 100% em entender o movimento sem interrupção
    LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)),
    LSTM(128, return_sequences=True, activation='relu'),
    LSTM(64, return_sequences=False, activation='relu'),
    
    # O Dropout fica SÓ AQUI, para evitar que ela decore a última etapa
    Dropout(0.2), 
    
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(ACTIONS.shape[0], activation='softmax')
])
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

callbacks = [
    # Salva automaticamente o melhor modelo (maior val accuracy)
    ModelCheckpoint('modelo_libras.h5', monitor='val_categorical_accuracy',
                    mode='max', save_best_only=True, verbose=1),
    # Para o treino se a val accuracy não melhorar por 40 epochs consecutivos
    EarlyStopping(monitor='val_categorical_accuracy', patience=40,
                  restore_best_weights=True, verbose=1),
]

print(f"\n--- TREINANDO COM {len(sequences)} SEQUÊNCIAS VÁLIDAS ---")
model.fit(X_train, y_train, epochs=300, batch_size=8,
          validation_data=(X_test, y_test), callbacks=callbacks)

print("\n✅ Modelo salvo com sucesso!")