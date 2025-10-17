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


def main():
    global last_mouse_x, last_mouse_y, last_left_click, last_right_click
    
    cap = cv2.VideoCapture(0) # Inicia a captura na câmera principal
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) # define a largura em 960px
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # define a altura em 540px

    with mp_hands.Hands(
        static_image_mode = False,
        max_num_hands = 1,
        min_detection_confidence = 0.6,
        min_tracking_confidence = 0.6
    ) as hands:
        
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                (points, hand_lms) = write_point(results, w, h)
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(thickness = 1, circle_radius = 2),
                    mp_drawing.DrawingSpec(thickness = 1)
                )
                
                ix, iy = points[INDEX_TIP]
                tx, ty = points[THUMB_TIP]
                mx, my = points[MIDDLE_TIP]
                
                cx, cy = mapping_screen(ix, iy, w, h)
                cx, cy = soften(cx, cy, last_mouse_x, last_mouse_y)
                pyautogui.moveTo(cx, cy)
                last_mouse_x, last_mouse_y = cx, cy
                
                d_thumb_index = distance((tx, ty), (ix, iy))
                d_index_middle = distance((ix, iy), (mx, my))
                
                now = time.time()
                
                norm_thresh = w / 15
                
                if d_thumb_index < norm_thresh * 0.5 and (now - last_left_click) > CLICK_COOLDOWN:
                    pyautogui.click()
                    last_left_click = now
                    
                if d_index_middle < norm_thresh * 0.45 and (now - last_right_click) > CLICK_COOLDOWN:
                    pyautogui.click(button = 'right')
                    last_right_click = now
                    
                cv2.putText(frame, 'Movimentacão: Indicador | Click Esquerdo: Polegar + Indicador | Click Direito: Indicador + Medio',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
                cv2.circle(frame, (ix, iy), 8, (0, 255,0), -1)
                cv2.circle(frame, (tx, ty), 8, (255, 255, 0), -1)
                cv2.circle(frame, (mx, my), 8, (0, 255, 255), -1)
                cv2.line(frame, (tx, ty), (ix, iy), (255, 255, 0), 2)
                cv2.line(frame, (ix, iy), (mx, my), (0, 255, 255), 2)
                
            cv2.imshow("Clique ESC para sair", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()