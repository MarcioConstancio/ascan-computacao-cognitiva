# Sistema de Reconhecimento Automático de Placas Automotivas (ALPR)

Este repositório apresenta a solução para o desafio da trilha de Computação Cognitiva focada em Visão Computacional, como parte do ASCAN, programa de Estágio do Instituto Atlântico. O objetivo do projeto é substituir um sistema legado de monitoramento de trânsito por um pipeline moderno de **Detecção de Objetos** e **Reconhecimento Óptico de Caracteres (OCR)**.

O sistema foi estruturado em duas etapas principais:
1. **Detecção de Placas:** Realizada utilizando o modelo **YOLOv8n** (Nano) ajustado via Transfer Learning.
2. **Reconhecimento de Texto:** Processado por meio da biblioteca **EasyOCR**, integrada a um pipeline de pré-processamento avançado com OpenCV (CLAHE) e filtros geométricos.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e reprodutibilidade:

```
|pasta-atual-do-projeto
├── doc/ # Contém as imagens usadas na escrita do notebook e do README
├── models/
│   ├───detector_placas.pt # Modelo YOLO8n treinado para detecção de placas
│   ├───detector_placas_v8n_otimizado.pt # Modelo YOLO8n treinado no Google Colab em GPU + Early Stopping.
│   └───detector_placas_v11.pt # Modelo YOLO11n treinado no Google Colab em GPU + Early Stopping.
├── src/
│   ├── download_data.py # Script para baixar o dataset do Kaggle
│   ├── prepare_data.py # Script para converter Pascal VOC em YOLO e dividir em treino/validação
│   ├── model_training.py # Script para treinar o modelo YOLO
│   ├── pipeline_alpr.py # Pipeline de inferência para produção
├── main.py # Script principal que abre aplicação Streamlit local capaz de processar imagens e exibir resultados
├── Desafio_Ascan_ComputacaoCognitiva.ipynb # Notebook com todo o desenvolvimento do projeto, análises e resultados obtidos.
├── README.md
├── data.yaml
├── pyproject.toml
├── uv.lock
└── .python-version
```
---

## Conjunto de dados e Processo de ETL

O projeto utiliza o dataset [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) do Kaggle, que possui 433 imagens com anotações no formato Pascal VOC (XML).

<center>
<figure>
  <img src="doc\Dif_pascal_yolo.png" width="50%" alt="Diferença entre Pascal VOC e YOLO">
  <figcaption>Tabela indicando parâmetros de bounding box em diferentes formatos. Disponível em: <a href="https://dragoneye.ai/blog/a-guide-to-bounding-box-formats/" target="_blank" rel="noopener">dragoneye</a> </figcaption>
</figure>
</center>


### Conversão Pascal VOC ──► YOLO
Como o formato nativo do dataset expressa as caixas em pixels absolutos `[xmin, ymin, xmax, ymax]`, o script `src/prepare_data.py` realiza a conversão matemática para o padrão YOLO `[classe_id, x_centro, y_centro, largura, altura]` normalizado entre `0.0` e `1.0`.

O script também divide os dados de forma aleatória e consistente na proporção **80% para treino** e **20% para validação**, gerando a árvore de pastas correta para o treinamento do YOLO.

---

## Como Executar o Projeto

Este projeto foi estruturado para atender a dois perfis de uso distintos: o **Usuário/Avaliador**, que deseja apenas testar a interface gráfica final, e o **Cientista de Dados**, que deseja reproduzir ou modificar o ciclo completo de desenvolvimento (download, preparação, treino e análise).

---

### 1. Instalação e Preparação do Ambiente

Antes de rodar qualquer uma das opções abaixo, clone o repositório e configure as dependências.

```bash
git clone git@github.com:MarcioConstancio/ascan-computacao-cognitiva.git
cd ascan-computacao-cognitiva
```

#### Opção A: Usando `uv` (Recomendado - Mais rápido)
Se você possui o `uv` instalado em sua máquina, o gerenciamento de ambiente é feito de forma automática:
```bash
# Sincroniza o ambiente virtual (.venv) e instala todas as dependências do uv.lock
uv sync
```

#### Opção B: Usando `pip` Tradicional
Caso prefira o gerenciamento de pacotes padrão do Python:
```bash
# Cria o ambiente virtual
python -m venv .venv

# Ativa o ambiente virtual
# No Windows (PowerShell):
.venv\Scripts\Activate.ps1
# No Linux/macOS:
source .venv/bin/activate

# Instala as dependências a partir do pyproject.toml
pip install --upgrade pip
pip install .
```

---

### 2. Como Executar como Usuário (Interface Web Streamlit)

Se o seu objetivo é apenas testar a ferramenta realizando inferências em novas imagens por meio de uma interface gráfica amigável:

1. **Garantir os Modelos:** Certifique-se de que o modelo treinado que deseja usar esteja localizado dentro da pasta `models/`.
2. **Executar a Aplicação:**

   * **Via `uv`:**
     ```bash
     uv run streamlit run main.py
     ```
   * **Via Python Tradicional (com venv ativo):**
     ```bash
     streamlit run main.py
     ```

3. **Utilização:** 
   O navegador abrirá automaticamente o endereço `http://localhost:8501`. Na barra lateral, configure as preferências de confiança e IOU. Na tela principal, envie um arquivo de imagem local ou cole um link direto da internet para visualizar os resultados do detector e os caracteres extraídos pelo OCR.

---

### 3. Como Executar como Cientista de Dados (Pipeline de Desenvolvimento)

Se você deseja reproduzir o pipeline do zero, treinar o modelo ou realizar experimentos avançados, siga os passos abaixo de forma sequencial na raiz do projeto:

#### Passo 1: Download Automatizado dos Dados
Baixa o dataset diretamente do Kaggle e o extrai para a raiz do diretório atual:

```bash
# Via uv
uv run python src/download_data.py

# Via venv ativo
python src/download_data.py
```

#### Passo 2: ETL e Preparação de Dados
Converte as anotações XML (Pascal VOC) em arquivos TXT (YOLO) e realiza a divisão de treino/validação (80%/20%), gerando a estrutura final em `data/yolo_dataset/`:
```bash
# Via uv
uv run python src/prepare_data.py
# Via venv ativo
python src/prepare_data.py
```

#### Passo 3: Treinamento do Modelo (Opcional)
Inicia o treinamento do modelo YOLO localmente (recomendado apenas se possuir GPU dedicada):
```bash
# Via uv
uv run python src/model_training.py
# Via venv ativo
python src/model_training.py
```

#### Passo 4: Exploração de Resultados e Métricas (Notebook)
Para realizar análises de erro qualitativas, avaliar gráficos de perda ou testar protótipos de pipeline de forma interativa:
1. Abra o seu editor (por exemplo, VS Code) na pasta do projeto.
2. Abra o arquivo `Desafio_Ascan_ComputacaoCognitiva.ipynb`.
3. Selecione o kernel do ambiente virtual `.venv` criado pelo `uv` ou pelo `venv`.
4. Execute as células sequencialmente para visualizar as métricas do YOLO, matrizes de confusão e testes com o pipeline de OCR.


## Conclusões

- O modelo yolov8.pt é capaz de detectar placas de veículos com uma boa precisão, mesmo em condições adversas, como iluminação ruim ou ângulos diferentes.
- A utilização do OCR permite extrair o texto em parte das placas detectadas, tornando possível seu uso para reconhecimento de placas automotivas. Embora ainda seja possível fazer melhorias no pré-processamento das imagens, como ajuste de brilho, contraste e nitidez, para aumentar a precisão do OCR.
- Outra opção seria utilizar modelos de OCR especializados em placas como [LPRNet](https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/lprnet/-?_lr=1), baseado em redes neurais convolucionais.

---

*Desafio desenvolvido para fins acadêmicos e de avaliação na trilha Computação Cognitiva.*