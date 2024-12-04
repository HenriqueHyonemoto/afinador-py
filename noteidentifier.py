import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

# Configurações do espectrograma
fig, ax = plt.subplots()
x = np.arange(0, CHUNK // 2)
line, = ax.plot(x, np.random.rand(CHUNK // 2))
ax.set_ylim(0, 1)
ax.set_xlim(0, CHUNK // 2)

# Função para atualizar o espectrograma
def update(frame):
    data = stream.read(CHUNK)
    samples = np.frombuffer(data, dtype=np.float32)
    fft_result = np.fft.fft(samples)
    fft_magnitude = np.abs(fft_result)
    line.set_ydata(fft_magnitude[:CHUNK // 2])
    return line,

# Cria a animação
ani = FuncAnimation(fig, update, blit=True, cache_frame_data=False)

plt.show()

# Fecha o stream e termina o PyAudio
stream.stop_stream()
stream.close()
p.terminate()
