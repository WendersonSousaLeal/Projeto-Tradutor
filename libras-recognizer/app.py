"""
app.py
=====================================================================
Aplicação web (Flask) que:
    1. Captura o feed da webcam no servidor.
    2. Processa cada frame com MediaPipe (via hand_utils.HandTracker).
    3. Classifica a letra de Libras com o RandomForest treinado.
    4. Aplica suavização temporal (votação por maioria) para evitar
       "piscar" entre letras erradas de frame a frame.
    5. Transmite o vídeo anotado para o navegador via MJPEG streaming
       e expõe um endpoint JSON com a letra atual.

Uso:
    python app.py
Depois acesse http://localhost:5000 no navegador.

Observação: o Flask serve o vídeo processado no SERVIDOR (onde a webcam
está conectada). Se você rodar isso em uma máquina remota, o navegador
mostrará a webcam DAQUELA máquina, não a do cliente.
"""

import os
from collections import Counter, deque

import cv2
import joblib
import numpy as np
from flask import Flask, Response, jsonify, render_template

from hand_utils import HandTracker, draw_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "libras_rf_model.pkl")

CONFIDENCE_THRESHOLD = 0.65  # predições abaixo disso são descartadas (reduz falsos positivos)
SMOOTHING_WINDOW = 8         # nº de predições recentes usadas na votação por maioria
STABILITY_MIN_RATIO = 0.5    # fração mínima de votos para "confirmar" uma letra

app = Flask(__name__)

# --- Carrega o modelo treinado (gerado por train_model.py) ---
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo não encontrado em '{MODEL_PATH}'. Rode collect_data.py e depois "
        "train_model.py antes de iniciar a aplicação web."
    )
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
label_encoder = bundle["label_encoder"]

tracker = HandTracker()
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Não foi possível acessar a webcam (índice 0).")

prediction_buffer = deque(maxlen=SMOOTHING_WINDOW)
current_state = {"letter": "-", "confidence": 0.0, "hand_visible": False}


def predict_letter(feature_vector: np.ndarray):
    """Retorna (letra, confiança). Se a confiança ficar abaixo do limiar,
    a letra retornada é None (o chamador decide o que fazer)."""
    probs = model.predict_proba([feature_vector])[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence

    letter = label_encoder.inverse_transform([best_idx])[0]
    return letter, confidence


def generate_frames():
    """Generator usado pelo Flask para o streaming MJPEG: lê a webcam,
    processa, desenha overlays e produz os bytes JPEG de cada frame."""
    global current_state

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        feature_vector, raw_landmarks = tracker.process(frame)
        hand_visible = raw_landmarks is not None

        if not hand_visible:
            # --- Nenhuma mão na tela: zera o buffer e o estado exibido ---
            prediction_buffer.clear()
            current_state = {"letter": "-", "confidence": 0.0, "hand_visible": False}
            cv2.putText(frame, "Nenhuma mao detectada", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            draw_landmarks(frame, raw_landmarks)
            letter, confidence = predict_letter(feature_vector)

            if letter is not None:
                prediction_buffer.append(letter)

            if prediction_buffer:
                voted_letter, votes = Counter(prediction_buffer).most_common(1)[0]
                stability = votes / len(prediction_buffer)
                if stability >= STABILITY_MIN_RATIO:
                    current_state = {
                        "letter": voted_letter,
                        "confidence": confidence,
                        "hand_visible": True,
                    }
            else:
                current_state = {"letter": "?", "confidence": 0.0, "hand_visible": True}

            # --- Overlay do resultado no próprio vídeo ---
            label = current_state["letter"]
            conf_pct = current_state["confidence"] * 100
            cv2.rectangle(frame, (10, 10), (280, 75), (0, 0, 0), -1)
            cv2.putText(frame, f"{label}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 3)
            cv2.putText(frame, f"{conf_pct:.0f}%", (120, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/current_letter")
def current_letter():
    """Endpoint usado pelo JS da página para atualizar o painel de leitura
    (independente do overlay já desenhado dentro do próprio vídeo)."""
    return jsonify(current_state)


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        camera.release()
        tracker.close()
