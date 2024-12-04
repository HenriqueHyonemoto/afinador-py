# pip install pyaudio aubio numpy

import pyaudio
import numpy as np
from aubio import source, pitch

# Configurações do microfone
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Inicializa o PyAudio
p = pyaudio.PyAudio()

# Abre o stream de áudio
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Configura o Aubio para detecção de pitch
tolerance = 0.8
pitch_o = pitch("yin", CHUNK, CHUNK, RATE)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

print("Capturando áudio... Pressione Ctrl+C para parar.")

try:
    while True:
        # Lê dados do microfone
        data = stream.read(CHUNK)
        samples = np.frombuffer(data, dtype=np.float32)

        # Detecta o pitch
        pitch_value = pitch_o(samples)[0]
        note = pitch_value if pitch_value > 0 else None

        if note:
            print(f"Nota detectada: {note:.2f} MIDI")

except KeyboardInterrupt:
    print("Captura de áudio encerrada.")

finally:
    # Fecha o stream e termina o PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
