"""
hand_utils.py
=====================================================================
Modulo compartilhado entre collect_data.py e app.py.

Responsabilidades:
  1. Encapsular o MediaPipe HandLandmarker (API "Tasks", que substituiu
     a antiga `mp.solutions.hands.Hands()` - a solução legada foi
     descontinuada pelo Google em favor desta nova API baseada em
     modelos .task).
  2. Extrair e normalizar os 21 landmarks (x, y, z) da mão, tornando-os
     invariantes a distância da câmera e posição na tela.
  3. Opcionalmente, calcular features de "velocidade" (delta entre
     frames) para ajudar a reconhecer letras com movimento (H, J, K, X, Z).
  4. Desenhar o esqueleto da mão sobre o frame para feedback visual.

Manter essa lógica em um único módulo garante que o MESMO pipeline de
extração/normalização seja usado tanto na coleta de dados quanto na
predição em tempo real - se eles divergirem, o modelo treinado recebe
features diferentes das que foram usadas no treino e a acurácia despenca.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------------------
# Configuração geral
# ---------------------------------------------------------------------------

# Caminho local do modelo de detecção de mãos do MediaPipe (baixado sob demanda)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

NUM_HANDS = 1  # o alfabeto manual de Libras é feito com uma mão só

# Liga/desliga as features de velocidade (posição atual - posição no frame anterior).
# Quando True, o vetor de features passa de 63 para 126 valores.
# Isso ajuda MUITO a distinguir letras com pequeno movimento (H, J, K, X, Z)
# de letras estáticas parecidas, sem precisar trocar de RandomForest para um
# modelo sequencial (LSTM/GRU). Veja a explicação completa na seção de
# "dicas de otimização" da resposta.
INCLUDE_VELOCITY = True

# Letras do alfabeto manual de Libras que envolvem movimento do pulso/dedos
# (confirmado com fontes sobre datilologia em Libras). Usado apenas para dar
# feedback ao usuário durante a coleta - não afeta o pipeline de features.
MOVEMENT_LETTERS = {"H", "J", "K", "X", "Z"}

# Conexões entre os 21 pontos da mão, para desenhar o "esqueleto".
# Essa constante é apenas uma lista estática de pares de índices, então
# continua disponível mesmo com a solução legada `mp.solutions.hands`
# descontinuada.
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def ensure_model_downloaded(path: str = MODEL_PATH) -> None:
    """Garante que o modelo .task do MediaPipe exista localmente, baixando-o
    automaticamente na primeira execução (requer internet apenas uma vez)."""
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[hand_utils] Modelo não encontrado. Baixando de {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    print(f"[hand_utils] Download concluído: {path}")


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    """
    Normaliza os 21 landmarks (shape (21, 3)) para torná-los invariantes a:
      - Translação: a posição da mão na tela não deve importar.
      - Escala: a distância do usuário até a câmera não deve importar.

    Deliberadamente NÃO removemos a rotação/orientação da mão: em Libras,
    a direção da palma é parte do significado do sinal (ex.: K, H e P usam
    configurações de dedos parecidas, mas se diferenciam pela orientação
    da mão). Uma normalização rotacional "completa" apagaria justamente a
    informação que distingue esses sinais.

    Passo 1 (translação): usamos o pulso (landmark 0) como origem (0,0,0),
    subtraindo suas coordenadas de todos os outros pontos.

    Passo 2 (escala): dividimos todos os pontos pela distância entre o
    pulso e a base do dedo médio (landmark 9). Essa distância funciona
    como uma "régua" proporcional ao tamanho da mão na imagem, que muda
    conforme o usuário se aproxima/afasta da câmera - dividir por ela
    cancela esse efeito.
    """
    wrist = points[0].copy()
    translated = points - wrist  # pulso vira a origem

    ref_distance = np.linalg.norm(translated[9])
    if ref_distance < 1e-6:
        ref_distance = 1e-6  # evita divisão por zero em casos degenerados

    scaled = translated / ref_distance
    return scaled.flatten().astype(np.float32)  # shape (63,)


class HandTracker:
    """Encapsula o MediaPipe HandLandmarker e o pipeline de features."""

    def __init__(self, model_path: str = MODEL_PATH, num_hands: int = NUM_HANDS,
                 include_velocity: bool = INCLUDE_VELOCITY):
        ensure_model_downloaded(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        self.include_velocity = include_velocity
        self._prev_normalized = None  # guarda o frame anterior p/ calcular velocidade
        self._frame_idx = 0

    @property
    def feature_size(self) -> int:
        return 126 if self.include_velocity else 63

    def process(self, frame_bgr: np.ndarray):
        """
        Processa um frame (formato BGR, como o OpenCV entrega) e retorna:
            feature_vector : np.ndarray | None  -> vetor pronto para o modelo
            raw_landmarks  : list | None         -> landmarks crus (p/ desenho)

        Retorna (None, None) quando nenhuma mão é detectada no frame.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # running_mode=VIDEO exige um timestamp crescente em milissegundos
        self._frame_idx += 1
        timestamp_ms = int(self._frame_idx * (1000 / 30))  # assume ~30 fps

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            self._prev_normalized = None  # sem mão -> zera referência de velocidade
            return None, None

        hand_landmarks = result.hand_landmarks[0]  # apenas a primeira mão detectada
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

        normalized = normalize_landmarks(points)  # shape (63,)

        if not self.include_velocity:
            return normalized, hand_landmarks

        if self._prev_normalized is None:
            velocity = np.zeros_like(normalized)
        else:
            velocity = normalized - self._prev_normalized
        self._prev_normalized = normalized

        feature_vector = np.concatenate([normalized, velocity])  # shape (126,)
        return feature_vector, hand_landmarks

    def close(self):
        self._landmarker.close()


def draw_landmarks(frame_bgr: np.ndarray, hand_landmarks, connections=HAND_CONNECTIONS):
    """Desenha o esqueleto da mão (pontos + conexões) diretamente sobre o frame."""
    h, w = frame_bgr.shape[:2]
    points_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in connections:
        cv2.line(frame_bgr, points_px[start_idx], points_px[end_idx], (255, 255, 255), 2)
    for x, y in points_px:
        cv2.circle(frame_bgr, (x, y), 4, (0, 200, 0), -1)

    return frame_bgr


def feature_column_names(include_velocity: bool = INCLUDE_VELOCITY):
    """Gera os nomes das colunas do CSV de dataset, na mesma ordem usada
    pelo HandTracker.process()."""
    cols = []
    for i in range(21):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    if include_velocity:
        for i in range(21):
            cols += [f"vx{i}", f"vy{i}", f"vz{i}"]
    return cols
