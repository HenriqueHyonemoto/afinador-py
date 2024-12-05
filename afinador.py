import pyaudio
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configurações do áudio
FORMATO = pyaudio.paInt16  # Formato de áudio representados como inteiros de 16 bits.
CANAIS = 1  # Número de canais
TAXA_AMOSTRAGEM = 44100  # Taxa de amostragem
TAMANHO_BUFFER = 1024  # Tamanho do buffer

LIMIAR_VOLUME = 1250  # MODIFICAVEL | Serve para definir volume mínimo de áudio detectável
MARGEM_ERRO = 2  # MODIFICAVEL | Margem de erro para a afinação

# Variáveis globais para armazenar a última nota, instrução e frequência detectadas
ultima_nota = None
ultima_instrucao = None
ultima_frequencia = None

# Função para capturar o áudio
def capturar_audio():
    # Inicializa a biblioteca pyaudio
    audio = pyaudio.PyAudio()

    # Abre uma captura de audio do Microfone
    stream = audio.open(format=FORMATO,
                        channels=CANAIS,
                        rate=TAXA_AMOSTRAGEM,
                        input=True,
                        frames_per_buffer=TAMANHO_BUFFER)
    frames = []

    # Loop para ler os frames de áudio do stream e armazená-los na lista `frames`
    for i in range(0, int(TAXA_AMOSTRAGEM / TAMANHO_BUFFER * 1)):  # MODIFICAVEL | Grava áudio de 1 em 1 segundo
        data = stream.read(TAMANHO_BUFFER)
        frames.append(data)

    # Encerrar a gravação
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Retorna o audio gravado por esse periodo (1s setado acima)
    return b''.join(frames)

# Função para processar o áudio e encontrar a frequência dominante
def processar_audio(dados_audio):
    # Processar um sinal de áudio, e converte para um formato manipulável em código
    array_audio = np.frombuffer(dados_audio, dtype=np.int16)
    N = len(array_audio)
    T = 1.0 / TAXA_AMOSTRAGEM
    yf = scipy.fft.fft(array_audio)
    xf = scipy.fft.fftfreq(N, T)[:N//2]

    # Calcular a amplitude média
    amplitude_media = np.mean(np.abs(array_audio))

    # Ignorar se a amplitude média estiver abaixo do limiar
    if amplitude_media < LIMIAR_VOLUME:
        return None, xf, np.abs(yf[:N//2])  

    # Encontrar a frequência dominante
    idx = np.argmax(np.abs(yf[:N//2]))
    freq = xf[idx]

    # Retorna os valores obtidos
    return freq, xf, np.abs(yf[:N//2])

# Função para mapear a frequência para a nota musical
def frequencia_para_nota(freq):
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # Todas as 12 Notas Musicais
    A4 = 440.0 # Define a nota LA4 com 440hz
    C0 = A4 * pow(2, -4.75) # Calcula a frequencia de C0 com base na frequencia de LA4
    nome = "" # Nome da nota (C D E F G A B etc)

    if freq == 0: # Ignorar caso não tenha som
        return None

    h = round(12 * np.log2(freq / C0)) # Calcula quantos semitons a frequencia esta acima de C0
    oitava = h // 12 # Calcula em qual oitava esta a nota (ex: agudo, medio, grave)
    n = h % 12 # Calcula qual o tom da nota dentro da oitava

    # Verifica se o índice da nota (n) está dentro do intervalo válido (correção de bug). 
    if n < 0 or n >= len(notas): 
        return None

    nome = notas[n] + str(oitava) # O nome da nota + a oitava (Ex A4, C2, F5)

    # Retorna dados obtidos (Nota + Posição da nota)
    return nome, h

# Função para determinar se a corda deve ser apertada ou afrouxada
def instrucoes_afinacao(freq, nota, frequencia_nota):
    diferenca = freq - frequencia_nota 
    if abs(diferenca) <= MARGEM_ERRO:
        return "Afinado"
    elif diferenca > MARGEM_ERRO:
        return "Afrouxe a corda"
    elif diferenca < -MARGEM_ERRO:
        return "Aperte a corda"

# Função para converter nota para frequência
def nota_para_frequencia(nota):
    notas = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] #Lista de Notas
    A4 = 440.0 # Definindo a frequencia de 440HZ (Padrão atual)
    C0 = A4 * pow(2, -4.75) #Define a primeira nota C0 com equação que calcula a partir do LA4
    nome_nota = nota[:-1]
    oitava = int(nota[-1])

    if nome_nota not in notas:
        return None

    indice_nota = notas.index(nome_nota)
    h = 12 * oitava + indice_nota
    return C0 * pow(2, h / 12)

# Função para atualizar o gráfico
def atualizar_grafico(frame):
    global ultima_nota, ultima_instrucao, ultima_frequencia
    dados_audio = capturar_audio()
    freq, xf, yf = processar_audio(dados_audio)

    plt.clf()
    plt.plot(xf, yf)
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')

    if freq is not None:
        info_nota = frequencia_para_nota(freq)
        if info_nota is None:
            # Se a nota não for reconhecida, continuar mostrando a última nota reconhecida
            if ultima_nota:
                plt.title(f'Última Frequência: {ultima_frequencia:.2f} Hz\nÚltima Nota: {ultima_nota}\nÚltima Instrução: {ultima_instrucao}')
            else:
                plt.title('Nota não reconhecida')
            return

        nota, h = info_nota
        frequencia_nota = nota_para_frequencia(nota)
        if frequencia_nota is None:
            # Se a nota não for reconhecida, continuar mostrando a última nota reconhecida
            if ultima_nota:
                plt.title(f'Última Frequência: {ultima_frequencia:.2f} Hz\nÚltima Nota: {ultima_nota}\nÚltima Instrução: {ultima_instrucao}')
            else:
                plt.title('Nota não reconhecida')
            return

        ultima_nota = f"{nota} ({frequencia_nota:.2f} Hz)"
        instrucao = instrucoes_afinacao(freq, nota, frequencia_nota)
        ultima_instrucao = instrucao
        ultima_frequencia = freq

        # Definir a cor do texto com base na instrução
        if instrucao == "Afinado":
            cor_texto = 'green'
        else:
            cor_texto = 'orange'

        plt.title(f'Frequência Dominante: {freq:.2f} Hz\nNota: {ultima_nota}\nInstrução: {instrucao}', color=cor_texto)
    else:
        if ultima_nota:
            plt.title(f'Última Frequência: {ultima_frequencia:.2f} Hz\nÚltima Nota: {ultima_nota}\nÚltima Instrução: {ultima_instrucao}')
        else:
            plt.title('Volume abaixo do limiar')

# Configura a animação do Matplotlib
fig, ax = plt.subplots()
ani = FuncAnimation(fig, atualizar_grafico, interval=1000)  # MODIFICAVEL, Atualiza o gráfico a cada 1 segundo

# Mostra a janela do Matplotlib
plt.show()
