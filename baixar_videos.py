"""
Baixa os vídeos de LIBRAS do YouTube e organiza em videos_baixados/SINAL/sinal.mp4
Usa yt-dlp. Para instalar: pip install yt-dlp
"""
import subprocess
import sys
from pathlib import Path

OUTPUT_BASE = Path("videos_baixados")

# Mapa: SINAL -> URL do YouTube
# Fonte principal: LibrasLab (vídeos de sinal único, qualidade consistente)
# Marcado com (*) onde foi usado outro canal por ausência no LibrasLab
VIDEOS = {
    "OI":          "https://www.youtube.com/watch?v=HYuNejbG8eo",
    "TCHAU":       "https://www.youtube.com/watch?v=KM4wNnXYTTs",
    "EU":          "https://www.youtube.com/watch?v=JMFyWhnZZXA",
    "NOME":        "https://www.youtube.com/watch?v=tIaoa5T6Luw",
    "SIM":         "https://www.youtube.com/watch?v=4_gsOiAvvoI",
    "NAO":         "https://www.youtube.com/watch?v=g5uLm2N0mY0",
    "GOSTAR":      "https://www.youtube.com/watch?v=Jc3IK1o7818",
    "AJUDA":       "https://www.youtube.com/watch?v=YCX7wF1aFhU",   # "AJUDAR" no LibrasLab
    "AMIGO":       "https://www.youtube.com/watch?v=SN_enrb0Lbs",
    "SURDO":       "https://www.youtube.com/watch?v=etvOu_qp_Gw",
    # (*) Outros canais — vídeos dedicados a um único sinal
    "OBRIGADO":    "https://www.youtube.com/watch?v=fr7pML3N0F4",
    "DESCULPA":    "https://www.youtube.com/watch?v=9k79Y9o3Xw0",
    "REPETIR":     "https://www.youtube.com/watch?v=edI0nceC-vg",
    "POR_FAVOR":   "https://www.youtube.com/watch?v=df4_7UMxvJ8",
    "BEM":         "https://www.youtube.com/watch?v=8f_hrn6EHhE",
    "ENTENDER":    "https://www.youtube.com/watch?v=kApN_DHhaJI",
    "NAO_ENTENDER":"https://www.youtube.com/watch?v=t77fuxgNv1M",
    "PRAZER":      "https://www.youtube.com/watch?v=gTuUJll9U_I",
}

def baixar(sinal, url):
    pasta = OUTPUT_BASE / sinal
    pasta.mkdir(parents=True, exist_ok=True)

    saida = pasta / f"{sinal.lower()}.mp4"
    if saida.exists():
        print(f"  [PULANDO] {sinal} — já existe")
        return

    print(f"  [BAIXANDO] {sinal} ...")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-check-certificate",
        "-f", "b[ext=mp4]/best[ext=mp4]/best",
        "-o", str(saida),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK] {sinal} salvo em {saida}")
    else:
        print(f"  [ERRO] {sinal}: {result.stderr.splitlines()[-1] if result.stderr else 'erro desconhecido'}")

if __name__ == "__main__":
    print(f"Baixando {len(VIDEOS)} vídeos...\n")
    for sinal, url in VIDEOS.items():
        baixar(sinal, url)
    print("\nPronto! Rode processador_videos.py para extrair os landmarks.")
