from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# # 1. Carrega o modelo pré-treinado (YOLOv8 Nano)
# model = YOLO('yolov8n.pt')

# # 2. Executa o treinamento
# # Ajuste o parâmetro 'epochs' e 'batch' conforme os recursos da sua máquina.
# results = model.train(
#     data='data.yaml',       # Arquivo de configuração
#     epochs=30,              # Número de épocas
#     imgsz=640,              # Redimensionamento padrão do YOLO
#     batch=16,               # Tamanho do lote (batch size)
#     device='cpu',           # Mude para '0' ou 'cuda' se tiver GPU Nvidia instalada
#     workers=2,              # Número de subprocessos para carregar dados
#     name='yolo_placas'      # Nome da pasta onde os pesos salvos serão guardados
# )



def mostrar_imagem_no_terminal(caminho_imagem, titulo):
    img = cv2.imread(caminho_imagem) # Carrega a imagem usando OpenCV
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converte de BGR para RGB para o matplotlib
        
        # Configura a janela de exibição
        plt.figure(figsize=(10, 6))
        plt.imshow(img_rgb)
        plt.title(titulo)
        plt.axis('off') 
        plt.show()      # Abre a janela física e pausa o script até que seja fechada
    else:
        print(f"Erro: Não foi possível carregar a imagem em '{caminho_imagem}'")

# 1. Exibe o Gráfico Geral de Resultados
print("--- Exibindo Evolução do Treinamento ---")
mostrar_imagem_no_terminal('runs/detect/yolo_placas/results.png', "Evolução do Treinamento")

# 2. Exibe a Matriz de Confusão
print("\n--- Exibindo Matriz de Confusão ---")
mostrar_imagem_no_terminal('runs/detect/yolo_placas/confusion_matrix.png', "Matriz de Confusão")