import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui

def main():
    print("=== Iniciando Treinamento do Modelo MNIST ===\n")

    # Carrega o dataset MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normaliza e ajusta o shape sem usar numpy (usando o próprio TF)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    # Constroi a CNN 
    model = tf.keras.Sequential([
        # Bloco Convolucional 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Bloco Convolucional 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Bloco Convolucional 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Camada de Saída com Dropout
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5), # Regularização antes da saída
        tf.keras.layers.Dense(10, activation='softmax') # 10 classes (0-9)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Treina com EarlyStopping (monitorando validação)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # Separa conjunto de validação explícito direto no fit (validation_split=0.2)
    model.fit(
        x_train, y_train,
        epochs=15, 
        validation_split=0.2, 
        callbacks=[early_stopping],
        verbose=1
    )

    # Exibi a acurácia de validação final no terminal
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nAcurácia de validação final: {val_acc:.4f}")

    # Salva o modelo treinado como "model.h5"
    model.save("model.h5")
    print("\nModelo salvo com sucesso!")

if __name__ == "__main__":
    main()