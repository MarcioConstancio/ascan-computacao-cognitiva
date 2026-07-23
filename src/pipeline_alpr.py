import easyocr
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


class ALPRPipelineProd:
    def __init__(self, detector_path):
        self.detector = YOLO(detector_path)
        # Inicializa o EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=False)
        # Define quais caracteres são válidos para uma placa automotiva
        self.char_allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


    ### FUNÇÃO PARA PRÉ-PROCESSAR A PLACA ##########
    def preprocess_plate(self, plate_crop):
            """
            Aplica técnicas de OpenCV para melhorar a legibilidade da placa antes do OCR.
            """
            # 1. Converter para escala de cinza
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # 2. Redimensionar (aumentar a imagem ajuda o OCR a identificar caracteres pequenos)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # 3. Suavização para reduzir ruído
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # 4. Binarização adaptativa (Otsu) para destacar os caracteres do fundo
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return thresh

    #### FUNÇÃO PARA PROCESSAR A IMAGEM ###### 
    def process_image(self, image_path, conf_threshold=0.25, iou=0.3):
        """
        Função responsável por aplicar o modelo de detecção, identificar a placa, converter as
        coordenadas relativas para coordenadas absolutas e cortar a região da placa com margem de 10%.
        Uma vez identificada a placa, esta passa pelo pré-processamento preprocess_plate para que o OCR
        possa fazer a detecção do texto.
        """

        img = cv2.imread(image_path)
        if img is None:
            print("Erro ao carregar a imagem.")
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # YOLO com iou=0.3 para 'forçar' o modelo a não detectar sobreposição de placas
        results = self.detector(img, conf=conf_threshold, iou=iou, verbose=False)[0]
        
        detections = []
        h, w, _ = img.shape
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = box.conf[0].item()
            
            # Margem de segurança de 10%
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = int(box_w * 0.10)
            pad_h = int(box_h * 0.10)
            
            crop_y1 = max(0, y1 - pad_h)
            crop_y2 = min(h, y2 + pad_h)
            crop_x1 = max(0, x1 - pad_w)
            crop_x2 = min(w, x2 + pad_w)
            
            plate_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_height, crop_width, _ = plate_crop.shape
            
            if plate_crop.size == 0:
                continue
                
            preprocessed_plate = self.preprocess_plate(plate_crop)

            ####### CHECKPOINT ############
            visual_plate = cv2.cvtColor(preprocessed_plate,cv2.COLOR_BGR2RGB)
            plt.imshow(visual_plate) 
            plt.show()
            ###############################
            
            # Executa o OCR
            # NOTA: O EasyOCR retorna as coordenadas no tamanho da imagem passada a ele.
            # Como redimensionamos a imagem por 2.5 no preprocess_plate, as coordenadas de retorno
            # estarão na escala da imagem pré-processada.
            ocr_results = self.reader.readtext(
                preprocessed_plate, 
                allowlist=self.char_allowlist, 
                decoder='greedy'
            )
            
            prep_h, prep_w = preprocessed_plate.shape[:2]
            
            valid_texts = []
            
            for res in ocr_results:
                bbox, text, ocr_conf = res # O formato do retorno é: [ [[x0,y0], [x1,y1], [x2,y2], [x3,y3]], text, confidence ]
                
                text = "".join([c for c in text.upper() if c != " "]) # Remove espaços
                
                # Calcula a altura da caixa de texto detectada pelo OCR
                ys = [pt[1] for pt in bbox]
                text_height = max(ys) - min(ys)
                
                # Altura Relativa do Texto (para ignorar letras de cidades/slogans pequenos)
                # Se a altura das letras detectadas for menor que 40% da altura total da placa, ignoramos.
                # Como a imagem foi escalada, calculamos a proporção em relação à altura da imagem pré-processada.
                proporcao_altura = text_height / prep_h
                
                if proporcao_altura < 0.40:
                    continue  # Descarte de texto pequeno (ex: cidades, molduras da placa)
                
                valid_texts.append((text, proporcao_altura))
            
            # Se houver mais de um bloco de texto que passou pelo filtro, priorizamos o bloco com maior altura relativa
            # (maior probabilidade de ser a placa principal)
            if valid_texts:
                # Ordena pelo segundo elemento da tupla (proporcao_altura) de forma decrescente
                valid_texts.sort(key=lambda x: x[1], reverse=True)
                print(valid_texts) # CHECKPOINT
                plate_text = valid_texts[0][0]
            else:
                plate_text = ""
                
            detections.append({
                'box': (x1, y1, x2, y2),
                'confidence': conf,
                'plate_text': plate_text,
                'crop': plate_crop,
                'preprocessed_crop': preprocessed_plate
            })
            
        return img_rgb, detections

    def plot_results(self, img_rgb, detections):
        # Converte de RGB de volta para exibição correta se necessário
        img_to_show = img_rgb
        
        plt.figure(figsize=(10, 6))
        plt.imshow(img_to_show)
        ax = plt.gca()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
            
            label = f"{det['plate_text']} ({det['confidence']:.2f})"
            plt.text(x1, y1 - 10, label, color='white', fontsize=12,
                     bbox=dict(facecolor='green', alpha=0.8, pad=2))
            
        plt.axis('off')
        plt.show()



if __name__ == "__main__":

    pipeline = ALPRPipelineProd(detector_path='models/detector_placas.pt')

    # Selecione um arquivo de imagem de validação para o teste
    imagens_teste = {
        'imagem_1': r"./yolo_dataset/images/val/Cars0.png",
        'imagem_2': r"C:\Users\marcio_constancio\Downloads\nissan-leaf-2022.jpg",
        'imagem_3': r"C:\Users\marcio_constancio\Downloads\placa-azul.webp",
        'imagem_4': r"C:\Users\marcio_constancio\Downloads\images.jpg",
        'imagem_5': r"C:\Users\marcio_constancio\Downloads\placa_amarela.jpg",
        'imagem_6': r"./yolo_dataset/images/val/Cars17.png",
        'imagem_7': r"./yolo_dataset/images/val/Cars18.png",
        'imagem_8': r"./yolo_dataset/images/val/Cars89.png",
        'imagem_9': r"./yolo_dataset/images/val/Cars224.png",
        'imagem_10': r"./yolo_dataset/images/val/Cars260.png",
        'imagem_11': r"./yolo_dataset/images/val/Cars277.png",
    }

    for imagem in imagens_teste.values():
        # Executa o pipeline
        resultado = pipeline.process_image(imagem, conf_threshold=0.41)

        if resultado:
            img_rgb, deteccoes = resultado
            pipeline.plot_results(img_rgb, deteccoes)
            
            # Exibe informações no console para depuração externa
            for idx, d in enumerate(deteccoes):
                print(f"Placa {idx+1}: {d['plate_text']} (Confiança Detector: {d['confidence']:.2f})")