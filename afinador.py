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

VOLUME_THRESHOLD = 300  # MODIFICAVEL | serve para definir volume minimo de audio detectavel
MARGEM_ERRO = 0.5  # MODIFICAVEL | Margem de erro para a afinação

# Variáveis globais para armazenar a última nota, instrução e frequência detectadas
last_note = None
last_instruction = None
last_frequency = None

# Função para capturar o áudio
def capture_audio():
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * 1)):  # MODIFICAVEL | grava audio de 1 em 1 segundo
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
        return None, xf, np.abs(yf[:N//2])  # Ignorar se a amplitude média estiver abaixo do limiar

    # Encontrar a frequência dominante
    idx = np.argmax(np.abs(yf[:N//2]))
    freq = xf[idx]

    return freq, xf, np.abs(yf[:N//2])

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

    return name, h

# Função para determinar se a corda deve ser apertada ou afrouxada
def tuning_instructions(freq, note, note_frequency):
    difference = freq - note_frequency
    if abs(difference) <= MARGEM_ERRO:
        return "Afinado"
    elif difference > MARGEM_ERRO:
        return "Afrouxe a corda"
    elif difference < -MARGEM_ERRO:
        return "Aperte a corda"

# Função para converter nota para frequência
def note_to_frequency(note):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)
    note_name = note[:-1]
    octave = int(note[-1])
    note_index = notes.index(note_name)
    h = 12 * octave + note_index
    return C0 * pow(2, h / 12)

# Função para atualizar o gráfico
def update_plot(frame):
    global last_note, last_instruction, last_frequency
    audio_data = capture_audio()
    freq, xf, yf = process_audio(audio_data)

    plt.clf()
    plt.plot(xf, yf)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')

    if freq is not None:
        note, h = frequency_to_note(freq)
        note_frequency = note_to_frequency(note)
        last_note = f"{note} ({note_frequency:.2f} Hz)"
        instruction = tuning_instructions(freq, note, note_frequency)
        last_instruction = instruction
        last_frequency = freq

        # Definir a cor do texto com base na instrução
        if instruction == "Afinado":
            text_color = 'green'
        else:
            text_color = 'orange'

        plt.title(f'Frequência Dominante: {freq:.2f} Hz\nNota: {last_note}\nInstrução: {instruction}', color=text_color)
    else:
        if last_note:
            plt.title(f'Última Frequência: {last_frequency:.2f} Hz\nÚltima Nota: {last_note}\nÚltima Instrução: {last_instruction}')
        else:
            plt.title('Volume abaixo do limiar')

# Configura a animação do Matplotlib
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, interval=1000)  # MODIFICAVEL, Atualiza o grafico a cada 1 segundo

# Mostra a janela do Matplotlib
plt.show()
