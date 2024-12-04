import pyaudio
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configurações do áudio
FORMAT = pyaudio.paInt16  # Formato de áudio
CHANNELS = 1  # Número de canais
RATE = 44100  # Taxa de amostragem
CHUNK = 1024  # Tamanho do buffer
VOLUME_THRESHOLD = 500  # Limiar de volume (ajuste conforme necessário)

# Variável global para armazenar a última nota detectada
last_note = None

# Função para capturar o áudio
def capture_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * 1)):  # Captura 1 segundo de áudio
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return b''.join(frames)

# Função para processar o áudio e encontrar a frequência dominante
def process_audio(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    N = len(audio_array)
    T = 1.0 / RATE
    yf = scipy.fft.fft(audio_array)
    xf = scipy.fft.fftfreq(N, T)[:N//2]

    # Calcular a amplitude média
    amplitude_mean = np.mean(np.abs(audio_array))

    if amplitude_mean < VOLUME_THRESHOLD:
        return None  # Ignorar se a amplitude média estiver abaixo do limiar

    # Encontrar a frequência dominante
    idx = np.argmax(np.abs(yf[:N//2]))
    freq = xf[idx]

    return freq

# Função para mapear a frequência para a nota musical
def frequency_to_note(freq):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)
    name = ""

    if freq == 0:
        return "N/A"

    h = round(12 * np.log2(freq / C0))
    octave = h // 12
    n = h % 12
    name = notes[n] + str(octave)

    return name

# Função para atualizar o gráfico
def update_plot(frame):
    global last_note
    audio_data = capture_audio()
    freq = process_audio(audio_data)

    if freq is not None:
        note = frequency_to_note(freq)
        last_note = note
        plt.clf()
        plt.title(f'Frequência Dominante: {freq:.2f} Hz\nNota: {note}')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 2000)
        plt.ylim(0, 1)
    else:
        plt.clf()
        if last_note:
            plt.title(f'Volume abaixo do limiar\nÚltima Nota: {last_note}')
        else:
            plt.title('Volume abaixo do limiar')
        plt.xlabel('Frequência (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(0, 2000)
        plt.ylim(0, 1)

# Configura a animação do Matplotlib
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, interval=1000)  # Atualiza a cada 1 segundo

# Mostra a janela do Matplotlib
plt.show()
