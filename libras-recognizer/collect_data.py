"""
collect_data.py
=====================================================================
Ferramenta de coleta de dados para o dataset de letras de Libras.

Captura o feed da webcam, extrai e normaliza os landmarks da mão a
cada frame (via hand_utils.HandTracker) e grava, sob demanda, uma
linha no CSV de dataset contendo: [label, feature_0, feature_1, ...].

CONTROLES (com a janela da webcam em foco):
    A-Z         -> seleciona a letra atual que está sendo gravada
    [ESPAÇO]    -> grava UMA amostra (frame atual) da letra selecionada
    [C]         -> alterna o modo de gravação CONTÍNUA (grava a cada
                   frame processado). Recomendado para as letras com
                   movimento: H, J, K, X, Z - faça o gesto completo
                   várias vezes com o modo contínuo ativo.
    [ESC]       -> encerra a coleta e fecha a janela

Uso:
    python collect_data.py
"""

import os
import csv

import cv2

from hand_utils import HandTracker, draw_landmarks, feature_column_names, MOVEMENT_LETTERS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks_dataset.csv")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    file_already_exists = os.path.isfile(CSV_PATH)

    tracker = HandTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Não foi possível acessar a webcam (índice 0). Verifique se ela está "
            "conectada e se nenhum outro programa está usando-a."
        )

    current_label = None
    continuous_mode = False
    samples_this_label = 0

    print("=== Coleta de Dados - Alfabeto Manual de Libras ===")
    print("A-Z: seleciona a letra | ESPAÇO: grava 1 amostra | C: modo contínuo | ESC: sair\n")

    # Abre o CSV em modo "append" - permite rodar o script várias vezes
    # (em sessões diferentes, com iluminação/ângulos diferentes) sem
    # perder os dados já coletados.
    with open(CSV_PATH, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_already_exists:
            writer.writerow(["label"] + feature_column_names())

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao capturar frame da webcam.")
                break

            frame = cv2.flip(frame, 1)  # espelha - fica mais natural para o usuário
            feature_vector, raw_landmarks = tracker.process(frame)

            hand_visible = raw_landmarks is not None
            if hand_visible:
                draw_landmarks(frame, raw_landmarks)

            # --- Overlay de status na tela ---
            cv2.putText(frame, f"Letra atual: {current_label or '-'}  |  Amostras: {samples_this_label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Modo: CONTINUO" if continuous_mode else "Modo: MANUAL (ESPACO p/ gravar)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            if current_label in MOVEMENT_LETTERS:
                cv2.putText(frame, "Letra com movimento - realize o gesto completo!",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

            if not hand_visible:
                cv2.putText(frame, "Nenhuma mao detectada", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Coleta de Dados - Libras (ESC para sair)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key in (ord("c"), ord("C")):
                continuous_mode = not continuous_mode

            elif ord("a") <= key <= ord("z"):
                current_label = chr(key).upper()
                samples_this_label = 0
                print(f"Letra selecionada: {current_label}")

            elif key == 32:  # barra de espaço
                if current_label is None:
                    print("Selecione uma letra (A-Z) antes de gravar uma amostra.")
                elif feature_vector is None:
                    print("Nenhuma mão detectada - amostra ignorada.")
                else:
                    writer.writerow([current_label] + feature_vector.tolist())
                    samples_this_label += 1

            # Gravação contínua: grava a cada frame enquanto ativa, útil para
            # capturar a variação natural do movimento em letras como H/J/K/X/Z
            if continuous_mode and current_label is not None and feature_vector is not None:
                writer.writerow([current_label] + feature_vector.tolist())
                samples_this_label += 1

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print(f"\nColeta finalizada. Dados salvos em: {CSV_PATH}")


if __name__ == "__main__":
    main()
