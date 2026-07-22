# Sistema de Reconhecimento Automático de Placas Automotivas (ALPR)

Este repositório contém a solução para o desafio da trilha de Computação Cognitiva focada em Visão Computacional, como parte ASCAN, programa de Estágio do Instituto Atlântico. O objetivo do projeto é substituir um sistema legado de monitoramento de trânsito por um pipeline moderno de **Detecção de Objetos** e **Reconhecimento Óptico de Caracteres (OCR)**.

O sistema foi estruturado em duas etapas principais:
1. **Detecção de Placas:** Realizada utilizando o modelo **YOLO8n** (Nano) ajustado via Transfer Learning.
2. **Reconhecimento de Texto:** Processado por meio da biblioteca **EasyOCR**, integrada a um pipeline de pré-processamento avançado com OpenCV (CLAHE) e filtros geométricos.

---

## 🛠️ Arquitetura do Pipeline

O pipeline de inferência foi projetado para ser resiliente a variações de iluminação, ruídos e inclinações de ângulo da câmera:

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e reprodutibilidade:

```
|pasta-atual-do-projeto
├── doc/ # Imagens usadas na escrita do notebook e do README
├── models/
│   ├───detector_placas.pt # Modelo YOLO8n treinado para detecção de placas
│   └───detector_placas_v11.pt # Modelo YOLO11n treinado no Google Colab em GPU + Early Stopping.
├── .python-version
├── data.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```
Ao rodar o notebook `Desafio_Ascan_ComputacaoCognitiva.ipynb`, o pipeline de inferência é executado, processando imagens de teste e exibindo os resultados de detecção e reconhecimento de placas. Nestas condições, as pastas `car-plate-detection`, `runs` e `yolo_dataset` serão criadas na raiz do projeto.

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

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório e Instalar Dependências
```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio
pip install -r requirements.txt
```

### 2. Baixar e Preparar os Dados
1. Baixe os dados do Kaggle e descompacte-os dentro da pasta `data/raw/` de forma que fiquem divididos em `data/raw/images` e `data/raw/annotations`.
2. Execute o script de preparação para estruturar o dataset do YOLO:
```bash
python src/prepare_data.py
```

### 3. Executar o Pipeline de Produção (Inferência)
Para processar uma imagem de teste e extrair o caractere da placa, utilize o script de pipeline:
```python
from src.pipeline_alpr import ALPRPipelineProducao

# Instancia o pipeline com o modelo treinado
pipeline = ALPRPipelineProducao(detector_path='models/detector_placas.pt')

# Processa a imagem
imagem_teste = "caminho/para/imagem.png"
img_rgb, deteccoes = pipeline.process_image(imagem_teste, conf_threshold=0.25)

for det in deteccoes:
    print(f"Placa Identificada: {det['plate_text']} (Confiança Detector: {det['confidence']:.2f})")
```

---

## 🧠 Análise Qualitativa de Erros e Qualidade do Dataset

Durante a etapa de validação cruzada e filtragem de falsos positivos, foram identificados comportamentos importantes que ajudam a entender as limitações do modelo e a complexidade do dataset:

### 1. Ruído de Rotulação (*Label Noise*)
* **Descoberta:** Na análise de imagens rotuladas como Falsos Positivos (ex: `Cars89`), observou-se que o modelo YOLO11n detectou corretamente placas de carros ao fundo. No entanto, o dataset original omitiu a anotação dessas placas secundárias no XML.
* **Impacto:** Isso demonstra que a capacidade de generalização do modelo é excelente, superando em alguns casos as anotações feitas por anotadores humanos, embora isso reduza artificialmente as métricas de validação teóricas.

### 2. Distratores Visuais e Redundância
* **Vazamento de dados:** Foram localizadas imagens duplicadas sob diferentes nomes no dataset (ex: `Cars18`, `Cars115` e `Cars161`), o que pode enviesar métricas se caírem em splits diferentes.
* **Falsos Positivos de Interface:** Em capturas de tela de sistemas operacionais, o modelo chegou a confundir o grupo de botões virtuais do Windows (minimizar, maximizar e fechar) como sendo uma placa. A similaridade geométrica (linha fina horizontal, alto contraste e elementos que mimetizam caracteres) explica essa confusão.

---

## 📈 Resultados Obtidos

* O treinamento do **YOLO11n** foi realizado no Google Colab sob uma GPU Tesla T4 utilizando a técnica de *Early Stopping* para prevenir o sobreajuste [1.2.6].
* O pipeline de pré-processamento **CLAHE + Remoção de Binarização Rígida** permitiu que o EasyOCR fizesse leituras estáveis mesmo em veículos posicionados de perfil (ângulos inclinados).
* O **Filtro de Altura Relativa** foi eficaz em remover ruídos textuais (como o nome de estados ou cidades nas placas), isolando apenas o código alfanumérico principal.

---

*Desafio desenvolvido para fins acadêmicos e de avaliação de Computação Cognitiva.*