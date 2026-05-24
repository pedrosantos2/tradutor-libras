import numpy as np
import os
import random
from pathlib import Path

# --- CONFIGURAÇÕES ---
DATA_PATH = Path("DATA")
ACTIONS = [
    "OI", "TCHAU", "EU", "NOME", "OBRIGADO", "SIM", "NAO",
    "POR_FAVOR", "DESCULPA", "BEM", "GOSTAR", "AJUDA",
    "ENTENDER", "NAO_ENTENDER", "REPETIR", "PRAZER", "AMIGO", "SURDO"
]
VARIACOES_POR_ARQUIVO = 10  # Para cada arquivo original, cria 10 novos

def aplicar_augment(dados_originais):
    """Aplica transformações espaciais nas coordenadas [x, y, z]"""
    dados_modificados = np.copy(dados_originais)
    
    # 1. ESCALA (Simula estar mais perto ou longe)
    escala = random.uniform(0.9, 1.1)
    
    # 2. TRANSLAÇÃO (Move levemente para os lados/cima)
    shift_x = random.uniform(-0.03, 0.03)
    shift_y = random.uniform(-0.03, 0.03)

    for frame in range(dados_modificados.shape[0]):
        for ponto in range(0, 126, 3):
            if dados_modificados[frame, ponto] != 0: # Ignora pontos não detectados
                # Aplica Escala e Translação
                dados_modificados[frame, ponto]   = (dados_modificados[frame, ponto] * escala) + shift_x
                dados_modificados[frame, ponto+1] = (dados_modificados[frame, ponto+1] * escala) + shift_y
                
                # 3. JITTER (Pequeno ruído/tremor natural)
                dados_modificados[frame, ponto]   += random.uniform(-0.002, 0.002)
                dados_modificados[frame, ponto+1] += random.uniform(-0.002, 0.002)

    return dados_modificados

print("🧬 Iniciando Expansão do Dataset para LIBRAS-SC...")

for action in ACTIONS:
    pasta_acao = DATA_PATH / action
    if not pasta_acao.exists():
        print(f"⚠️ Pasta {action} não encontrada. Pulando...")
        continue
    
    arquivos_originais = [f for f in os.listdir(pasta_acao) if f.endswith('.npy') and not f.startswith('aug_')]
    print(f"📁 Processando {len(arquivos_originais)} arquivos originais em: {action}")

    for nome_arq in arquivos_originais:
        caminho_arq = pasta_acao / nome_arq
        dados_base = np.load(caminho_arq)

        for i in range(VARIACOES_POR_ARQUIVO):
            dados_novos = aplicar_augment(dados_base)
            novo_nome = f"aug_{i}_{nome_arq}"
            np.save(pasta_acao / novo_nome, dados_novos)

print("\n✅ Dataset expandido com sucesso! Agora as 3 pastas estão prontas para o treino.")