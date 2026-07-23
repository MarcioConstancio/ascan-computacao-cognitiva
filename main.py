import streamlit as st
import os
import requests
import cv2
from PIL import Image
from src.pipeline_alpr import ALPRPipelineProd

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Sistema ALPR - Detecção e OCR",
    page_icon="🚗",
    layout="wide"
)

# 1. Carregamento do Pipeline com Cache do Streamlit (Garante carregamento único do modelo)
@st.cache_resource
def carregar_pipeline():
    # Caminho padrão dos melhores pesos obtidos
    caminho_modelo = "models/detector_placas.pt"
    
    if not os.path.exists(caminho_modelo):
        st.error(f"Pesos do modelo não encontrados. Certifique-se de que existe um arquivo .pt' no diretório.")
        st.stop()
        
    return ALPRPipelineProd(detector_path=caminho_modelo)

# Inicializa o pipeline
pipeline = carregar_pipeline()

# Título da Aplicação
st.title("🚗 Reconhecimento Automático de Placas (ALPR)")
st.write("Ferramenta para detecção e reconhecimento de caracteres em placas automotivas utilizando YOLO e EasyOCR.")

# Layout: Menu lateral para configurações de parâmetros do modelo
st.sidebar.header("Configurações do Modelo")
conf_threshold = st.sidebar.slider("Limiar de Confiança do YOLO",
                                   min_value=0.1,
                                   max_value=1.0,
                                   value=0.25,
                                   step=0.05,
                                   help="Define a certeza mínima que o modelo precisa ter para desenhar a caixa na placa. Valores maiores reduzem os falsos positivos."
                                   )
iou_threshold = st.sidebar.slider("Limiar de IOU (NMS) para evitar duplicadas",
                                  min_value=0.1,
                                  max_value=1.0,
                                  value=0.40,
                                  step=0.05,
                                  help=" Define o percentual máximo de sobreposição entre predições. Valores baixos minimizam caixas duplicadas."
                                  )

# Opções de entrada do usuário
opcao_entrada = st.radio("Escolha o método de envio da imagem:", ("Upload de Arquivo Local", "Inserir URL/Link da Imagem"))

# Variável para armazenar o caminho temporário da imagem
caminho_temp = "temp_input.png"
imagem_pronta = False

# 2. Processamento da Entrada
if opcao_entrada == "Upload de Arquivo Local":
    arquivo_upload = st.file_uploader("Selecione uma imagem (PNG, JPG, JPEG, WEBP)", type=["png", "jpg", "jpeg","webp"])
    if arquivo_upload is not None:
        try:
            image = Image.open(arquivo_upload)
            image.save(caminho_temp)
            imagem_pronta = True
        except Exception as e:
            st.error(f"Erro ao ler arquivo enviado: {e}")

else:
    url_imagem = st.text_input("Cole aqui o URL direto da imagem:")
    if url_imagem:
        try:
            # Faz o download da imagem via HTTP
            resposta = requests.get(url_imagem, stream=True, timeout=10)
            if resposta.status_code == 200:
                with open(caminho_temp, 'wb') as f:
                    f.write(resposta.content)
                imagem_pronta = True
            else:
                st.error(f"Erro ao acessar o link (Status Code: {resposta.status_code}). Certifique-se de que o URL aponta diretamente para uma imagem.")
        except Exception as e:
            st.error(f"Falha na requisição da imagem: {e}")

# 3. Execução do Pipeline e Exibição de Resultados
if imagem_pronta:
    if st.button("Executar Inferência"):
        with st.spinner("Processando imagem (Executando YOLO e OCR)..."):
            # Executa o processamento do pipeline ALPR
            # Retorna img_rgb processada e a lista de dicionários com as detecções
            resultado = pipeline.process_image(
                caminho_temp, 
                conf_threshold=conf_threshold, 
                iou=iou_threshold
            )
            
            if resultado:
                img_rgb, deteccoes = resultado
                
                # Interface organizada em 2 Colunas (Esquerda: Detecção / Direita: Resultados do OCR)
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Resultado Visual da Detecção")
                    
                    # Desenha as caixas e legendas na imagem para exibição no Streamlit
                    # Fazemos uma cópia para não alterar a imagem de origem
                    img_desenho = cv2.imread(caminho_temp)
                    
                    for det in deteccoes:
                        x1, y1, x2, y2 = det['box']
                        cv2.rectangle(img_desenho, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{det['plate_text']} ({det['confidence']:.2f})"
                        cv2.putText(img_desenho, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Exibe a imagem final convertendo BGR para RGB
                    st.image(cv2.cvtColor(img_desenho, cv2.COLOR_BGR2RGB), width="stretch")
                    
                with col2:
                    st.subheader("Extração de Caracteres (OCR)")
                    
                    if not deteccoes:
                        st.warning("Nenhuma placa de veículo foi localizada nesta imagem.")
                    else:
                        for idx, det in enumerate(detections_list := deteccoes):
                            st.write(f"### Placa {idx+1}")
                            
                            # Exibe o recorte original da placa gerado pelo YOLO
                            crop_rgb = cv2.cvtColor(det['crop'], cv2.COLOR_BGR2RGB)
                            st.image(crop_rgb, caption=f"Recorte da Placa {idx+1}", width="stretch")
                            
                            # Exibe a imagem binarizada de entrada do OCR
                            st.image(det['preprocessed_crop'], caption="Entrada Processada do OCR", channels="GRAY", width="stretch")
                            
                            # Exibe o texto e confiança obtidos
                            if det['plate_text']:
                                st.success(f"**Texto Reconhecido:** `{det['plate_text']}`")
                            else:
                                st.warning("**Texto Reconhecido:** `Falha na leitura` (Nenhum caractere válido identificado)")
                                
                            st.info(f"**Confiança do Detector:** {det['confidence']:.2%}")
                            st.write("---")
            
            # Limpa o arquivo temporário do disco após o processamento
            if os.path.exists(caminho_temp):
                os.remove(caminho_temp)