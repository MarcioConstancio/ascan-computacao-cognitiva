# Definição dos caminhos dos dados atuais
import os
import xml.etree.ElementTree as ET
import glob
from sklearn.model_selection import train_test_split
import shutil
import yaml

ORIGEM_IMG = "./car-plate-detection/images"
ORIGEM_XML = "./car-plate-detection/annotations"
DESTINO_BASE = "./yolo_dataset"

########## CRIAÇÃO DAS PASTAS NO FORMATO YOLO ###########################
# Estrutura de pastas exigida pelo YOLO
pastas_yolo = {
    'train_images': os.path.join(DESTINO_BASE, 'images', 'train'),
    'val_images': os.path.join(DESTINO_BASE, 'images', 'val'),
    'train_labels': os.path.join(DESTINO_BASE, 'labels', 'train'),
    'val_labels': os.path.join(DESTINO_BASE, 'labels', 'val')
}

# Criando as pastas de destino de acordo com a estrutura Yolo
for pasta in pastas_yolo.values():
    os.makedirs(pasta, exist_ok=True)
#########################################################################

########## FUNÇÃO DE CONVERSÃO XML (PASCAL VOC) -> TXT (YOLO)  ##########
def converter_xml_para_yolo(xml_path):
    try: #Parsing do arquivo XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Obtém dimensões da imagem
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        yolo_linhas = []
        
        # Itera sobre as tags object para confirmar se a classe se chama license
        for obj in root.iter('object'):
            classe = obj.find('name').text
            if classe != 'licence':
                continue
            
            classe_id = 0  # Classe única, então definimos ID = 0
            
            # Acessa a tag <bndbox> para coletar os valores das coordenadas (xmin,ymin) e (xmax,ymax) do formato PASCAL VOC
            box = obj.find('bndbox')
            xmin = float(box.find('xmin').text)
            ymin = float(box.find('ymin').text)
            xmax = float(box.find('xmax').text)
            ymax = float(box.find('ymax').text)
            
            # Converte as coordenadas de Pascal para Yolo
            x_centro = (xmin + xmax) / 2.0 / w
            y_centro = (ymin + ymax) / 2.0 / h
            largura_box = (xmax - xmin) / w
            altura_box = (ymax - ymin) / h
            
            # Restringe os valores para ficarem estritamente entre 0.0 e 1.0. Exemplo, caso alguns arredondamento
            # resulte em 1.0001 ou -0.0001, o Yolo retorna um erro e descarta a imagem durante o treinamento.
            x_centro = min(max(x_centro, 0.0), 1.0)
            y_centro = min(max(y_centro, 0.0), 1.0)
            largura_box = min(max(largura_box, 0.0), 1.0)
            altura_box = min(max(altura_box, 0.0), 1.0)
            
            # Cria as linhas para serem salvas no padrão Yolo e salva na lista yolo_linhas
            yolo_linhas.append(f"{classe_id} {x_centro:.6f} {y_centro:.6f} {largura_box:.6f} {altura_box:.6f}")
            
        return yolo_linhas # Retorna uma lista de linhas no formato Yolo
    except Exception as e:
        print(f"Erro ao ler {xml_path}: {e}")
        return None
#########################################################################

########## MAPEAMENTO DOS PARES XML E IMAGENS EXISTENTES ################
# Coleta os caminhos de todos os arquivos da pasta car-plate-detection/annotations/ que contém a extensão .xml
arquivos_xml = sorted(glob.glob(os.path.join(ORIGEM_XML, "*.xml")))
pares_validos = []

#Extrai o nome do arquivo e gera o nome base da figura correspondente. Por exemplo:
#a partir do caminho car-plate-detection/annotations/Cars132.xml é  gerado o caminho da imagem
#correspondente como car-plate-detection/images/Cars132.png. Se este caminho existir, adiciono o par de caminhos
#à lista de pares validos. Ex: (car-plate-detection/annotations/Cars132.xml,car-plate-detection/images/Cars132.png).
#Por fim, imprimo o total de pares válidos para que possam conferir com o numero total de registros. 
for xml_path in arquivos_xml:
    nome_base = os.path.splitext(os.path.basename(xml_path))[0]
    caminho_img = os.path.join(ORIGEM_IMG, f"{nome_base}.png")
    
    if os.path.exists(caminho_img):
        pares_validos.append((caminho_img, xml_path))

print(f"Total de amostras identificadas: {len(pares_validos)}")
#########################################################################

########## DIVISAO DO CONJUNTO EM TREINO (80%) E VALIDAÇÃO (20%) ########
treino_pares, val_pares = train_test_split(
    pares_validos, 
    train_size=0.8, 
    random_state=42
)
#########################################################################

########## FUNÇÃO PARA COPIAR AS IMAGENS PARA AS PASTAS CORRETAS E SALVAR OS ARQUIVOS TXT ################
def copiar_e_salvar(pares, split_nome):
    """
    Copia as imagens da pasta original para as pastas \images\train e \images\val. Cria os arquivos .txt correspondentes
    nas pastas \label\\train e \label\val.
    pares: lista de tuplas contendo os pares (caminho-imagem, caminho-xml). Gerada a partir do train_test_split acima
    split_nome: nome da pasta de destino que está sendo criada 'train','val' ou 'test'.
    """
    #Retorna o caminho das imagens de destino, baseado no que foi declarado no dicionário pastas_yolo
    pasta_img_dst = pastas_yolo[f'{split_nome}_images']
    pasta_lbl_dst = pastas_yolo[f'{split_nome}_labels']
    
    # Para cada par de arquivos, copia a imagem para o destino e cria o arquivo .txt com o mesmo nome na pasta labels
    for img_path, xml_path in pares:
        nome_base = os.path.splitext(os.path.basename(xml_path))[0]
        
        # Copia a imagem para o destino
        shutil.copy(img_path, os.path.join(pasta_img_dst, f"{nome_base}.png"))
        
        # Converte o XML correspondente e grava o TXT
        linhas_yolo = converter_xml_para_yolo(xml_path)
        if linhas_yolo:
            with open(os.path.join(pasta_lbl_dst, f"{nome_base}.txt"), 'w') as f:
                f.write('\n'.join(linhas_yolo))
#########################################################################

# Executa a distribuição
copiar_e_salvar(treino_pares, 'train')
copiar_e_salvar(val_pares, 'val')

print(f"Dados processados com sucesso!")
print(f"Treino: {len(treino_pares)} imagens em {pastas_yolo['train_images']}")
print(f"Validação: {len(val_pares)} imagens em {pastas_yolo['val_images']}")

### CONFIGURANDO ARQUIVO YAML
# Estrutura do arquivo de configuração para o YOLO
print("Criando o arquivo data.yaml...")

config_yolo = {
    'path': "yolo_dataset", # Caminho relativo para a pasta yolo_dataset, onde estão as imagens e anotações (labels)
    'train': 'images/train', #Caminho para as imagens de treinamento
    'val': 'images/val',
    'names': {
        0: 'licence'
    }
}

# Salvando no arquivo data.yaml
with open('data.yaml', 'w') as f:
    yaml.dump(config_yolo, f, default_flow_style=False)

print("Arquivo data.yaml criado com sucesso!")