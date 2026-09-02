# Leitor de Libras em Tempo Real

Sistema web para reconhecimento do alfabeto manual de Libras via webcam,
usando MediaPipe (extração de landmarks da mão) + RandomForest (classificação)
+ Flask (interface web com streaming de vídeo).

## Estrutura do projeto

```
libras-recognizer/
├── app.py                    # Servidor Flask (streaming + predição em tempo real)
├── collect_data.py           # Script de coleta de dados via webcam
├── train_model.py            # Script de treinamento do RandomForest
├── hand_utils.py             # Módulo compartilhado: MediaPipe + normalização
├── requirements.txt
├── README.md
├── data/
│   └── landmarks_dataset.csv     (gerado por collect_data.py)
├── models/
│   ├── hand_landmarker.task      (baixado automaticamente na 1ª execução)
│   └── libras_rf_model.pkl       (gerado por train_model.py)
└── templates/
    └── index.html             # Interface web (viewfinder + painel de leitura)
```

## Como o pipeline funciona

1. **Captura**: o OpenCV lê os frames da webcam.
2. **Extração**: o MediaPipe HandLandmarker detecta a mão e retorna 21
   pontos (x, y, z) por frame.
3. **Normalização** (`hand_utils.normalize_landmarks`): os pontos são
   deslocados para que o pulso vire a origem (0,0,0) e depois escalados
   pela distância pulso → base do dedo médio. Isso torna o reconhecimento
   independente de onde a mão está na tela e de quão perto/longe da câmera
   o usuário está. A orientação da mão é preservada de propósito, pois em
   Libras ela é parte do significado do sinal.
4. **Velocidade (opcional, ligado por padrão)**: para ajudar a reconhecer
   as letras com movimento (H, J, K, X, Z), cada frame também guarda a
   diferença entre a posição normalizada atual e a do frame anterior,
   dobrando o vetor de features de 63 para 126 valores.
5. **Classificação**: o vetor de features é passado para o
   `RandomForestClassifier` treinado, que retorna a letra + a confiança.
6. **Suavização**: a aplicação web faz uma votação por maioria entre as
   últimas 8 predições antes de exibir uma letra, evitando "piscar" entre
   letras erradas de frame a frame.

## Guia de execução

Veja o passo a passo detalhado abaixo (mesmos comandos, em ordem).

```bash
# 1. Criar e ativar um ambiente virtual
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Coletar dados de treinamento (repita para cada letra do alfabeto)
python collect_data.py

# 4. Treinar o modelo com os dados coletados
python train_model.py

# 5. Rodar a aplicação web
python app.py
# Acesse http://localhost:5000 no navegador
```

## Dicas de otimização para letras com movimento/variação

- **Features de velocidade**: já habilitadas por padrão em `hand_utils.py`
  (`INCLUDE_VELOCITY = True`). Elas dão ao RandomForest um sinal direto de
  "quanto e para onde" a mão se moveu entre dois frames, sem precisar
  trocar para um modelo sequencial (LSTM/GRU) para capturar movimento.
- **Coleta em modo contínuo**: para H, J, K, X e Z, use a tecla `C` em
  `collect_data.py` e realize o gesto completo várias vezes - isso captura
  a trajetória do movimento em várias amostras, não só a pose final.
- **Não normalize a rotação**: letras como K/H/P ou G/Q têm configurações
  de dedos parecidas e se diferenciam pela orientação da mão. Uma
  normalização que remove rotação apagaria essa diferença.
- **Suavização temporal na predição**: a votação por maioria em `app.py`
  (`SMOOTHING_WINDOW`, `STABILITY_MIN_RATIO`) reduz o efeito de frames
  ruidosos isolados durante o movimento.
- **Limiar de confiança**: `CONFIDENCE_THRESHOLD` em `app.py` descarta
  predições em que o modelo está pouco confiante, em vez de mostrar
  qualquer palpite.
- **Quantidade e diversidade de dados**: colete pelo menos 150-300 amostras
  por letra, variando distância até a câmera, ângulo da mão, iluminação e,
  se possível, mais de uma pessoa sinalizando. `train_model.py` avisa no
  console quando alguma letra tem poucas amostras.
- **class_weight="balanced"**: já usado em `train_model.py` para compensar
  letras com menos amostras do que outras.
- **Feature importance**: `train_model.py` imprime as 10 features mais
  importantes após o treino - útil para verificar se o modelo está usando
  landmarks que fazem sentido (ex.: ponta dos dedos) e não ruído.
- **Limite do RandomForest**: ele classifica cada frame de forma
  independente. Para sinais com movimento mais longo/complexo (além do
  alfabeto manual, por exemplo palavras inteiras em Libras), o próximo
  passo natural seria um modelo sequencial (ex.: LSTM sobre uma janela de
  frames), mas isso sai do escopo do RandomForest.

## Tratamento de "nenhuma mão visível"

Tanto em `collect_data.py` (overlay "Nenhuma mao detectada" + amostras
ignoradas) quanto em `app.py` (buffer de suavização é limpo e o painel
mostra "Aguardando mão no quadro...") o sistema trata explicitamente a
ausência de mão, em vez de tentar prever uma letra a partir de dados
inexistentes.
