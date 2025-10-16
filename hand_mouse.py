import cv2
import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

import mediapipe as mp
mp_hands = mp.solutions.hands # Modelo de detecção/rastreamento de mãos.
mp_drawing = mp.solutions.drawing_utils # Desenha a "malha" para debug visual.

# Variáveis para indicar cada ponto da mão

THUMB_TIP = 4 # Polegar
INDEX_TIP = 8 # Indicador
INDEX_PIP = 6 # Junta Indicador
MIDDLE_TIP = 12 # Médio

# Capturando tamanho da tela e paraâmetros do controle

SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 0.25 # Utilizada para suavizar o movimento do mouse. Evitando tremdeiras
MARGIN = 0.15 # Diminui 15% de cada lado do quadro da Webcam. Reduzindo cliques involuntários.
CLICK_COOLDOWN = 0.25 # Tempo para evitar vários cliques por um único gesto.

#  Variáveis de State

last_mouse_x = last_mouse_y = None
last_left_click = last_right_click = 0.0

# Funções de indentificação e parametrização da mão.

def write_point(results, w, h): # Função que retorna os pontos em pixels para desenha a malha (debug).
    if not results.multi_hand_landmarks:
        return None
    hand = results.multi_hand_landmarks[0]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmark]
    return points, hand

def distance (p1, p2): # Função para calcular a distância Euclidiana entre os dedos.
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def mapping_screen (x, y, frame_w, frame_h): # Reduz as bordas e interpola linearmente o espeça da câmera para espaço em tela.
    x_min = frame_w * MARGIN
    x_max = frame_w * (1 - MARGIN)
    y_min = frame_h * MARGIN
    y_max = frame_h * (1 - MARGIN)
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    scr_x = np.interp(x, [x_min, x_max], [0, SCREEN_W])
    src_y = np.interp(y, [y_min, y_max], [0, SCREEN_H])
    return scr_x, src_y

def soften (nx, ny, lx, ly, factor = SMOOTHING): # Função para aproximar o novo ponto do anterior
    if lx is None or ly is None:
        return nx, ny
    sx = lx + factor * (nx - lx)
    sy = ly + factor * (ny - ly)
    return sx, sy





