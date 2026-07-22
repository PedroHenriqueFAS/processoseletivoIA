import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui

def main():
    # Garanti que os caminhos funcionem independente de onde o script for chamado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    h5_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    # Carrega o modelo treinado em "model.h5"
    model = tf.keras.models.load_model(h5_path)

    # Converte para TensorFlow Lite usando tf.lite.TFLiteConverter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Aplica uma técnica de otimização
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Salva o resultado como "model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

if __name__ == "__main__":
    main()