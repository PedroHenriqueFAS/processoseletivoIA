# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo



---
## 📝 Relatório do Candidato

👤 **Nome Completo:** Pedro Henrique Ferreira Amorim da Silva

### 1️⃣ Resumo da Arquitetura do Modelo
A arquitetura implementada é uma Rede Neural Convolucional (CNN) focada em extração eficiente de características para Edge AI. O modelo é composto por 3 blocos convolucionais sequenciais. Cada bloco contém uma camada `Conv2D` (com 32, 64 e 128 filtros, respectivamente), seguida por uma camada de `BatchNormalization` (para estabilizar a distribuição dos dados no treinamento) e uma `MaxPooling2D` (redução de dimensionalidade). 

A camada de saída conta com um `Flatten`, seguido de um `Dropout` com taxa de 0.5, uma escolha de hiperparâmetro técnica essencial para mitigar o overfitting, forçando a rede a generalizar os traços dos dígitos em vez de memorizar os dados de treino. A saída é uma camada `Dense` com 10 neurônios (ativação `softmax`). A estratégia de validação utilizou um split de 20% (`validation_split=0.2`), e o treinamento foi monitorado por um `EarlyStopping` (paciência = 3), interrompendo o ciclo na época 12 para garantir a retenção dos pesos com maior capacidade de generalização.

### 2️⃣ Bibliotecas Utilizadas
- **TensorFlow / Keras (v2.21.0):** Framework principal para construção da arquitetura matemática, treinamento (`tf.keras`) e conversão para microcontroladores/mobile (`tf.lite`).
- **OS (Python Standard Library):** Utilizada para o gerenciamento dinâmico de caminhos absolutos e relativos do sistema operacional.
- **NumPy (v2.2.5):** Biblioteca fundamental utilizada para manipulação eficiente de arrays, pré-processamento de dados e operações numéricas sobre as matrizes das imagens do dataset.

### 3️⃣ Técnica de Otimização do Modelo
A técnica aplicada em `optimize_model.py` foi a **Quantização de Faixa Dinâmica (Dynamic Range Quantization)**. Através da configuração `tf.lite.Optimize.DEFAULT` no TFLiteConverter, os pesos do modelo original (que ocupavam 32 bits em ponto flutuante - `float32`) foram comprimidos para números inteiros de 8 bits (`int8`). Essa técnica otimiza o uso de memória RAM e acelera a latência de inferência em processadores de borda (Edge AI) praticamente sem perda de acurácia.

### 4️⃣ Resultados Obtidos
- **Acurácia de Validação Final:** 99,03% 
- **Tamanho do arquivo original (`model.h5`):** 1,14 MB
- **Tamanho do arquivo otimizado (`model.tflite`):** 103 KB
- **Comparação Numérica:** Houve uma redução extrema de footprint. O modelo otimizado (`.tflite`) ficou aproximadamente **11 vezes menor** que o original (`.h5`), caindo de 1,14 MB para apenas 103 KB. Isso valida o sucesso da quantização para viabilizar o embarque da IA em dispositivos com restrição severa de armazenamento.

### 5️⃣ Comentários Adicionais
Uma decisão técnica vital tomada durante o projeto foi a estruturação de caminhos relativos usando a biblioteca `os`. Inicialmente, o uso de caminhos absolutos gerou um problema de *encoding* no sistema operacional hospedeiro (Windows), onde caracteres especiais no diretório-raiz impediam a leitura do artefato pelo interpretador C++ interno do TF Lite durante a otimização. A resolução através do `os.path.join` garantiu a isolação do ambiente e tornou o script reproduzível em qualquer máquina ou sistema de CI/CD (como o GitHub Actions). 

### 6️⃣ Exemplo de Inferência
**Saída do Terminal:**
```text
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
````
O modelo obteve 100% de precisão nas 5 amostras aleatórias do conjunto de teste (inédito). É particularmente interessante observar o acerto na Amostra 1 (dígito '7') e na Amostra 3 (dígito '1'). Esses dois numerais frequentemente compartilham características de traço vertical inclinado que costumam confundir modelos mais rasos. O acerto perfeito com o modelo pós-quantização (rodando em int8) demonstra que as camadas de convolução conseguiram extrair features robustas o suficiente para diferenciar essas classes com clareza na inferência.

