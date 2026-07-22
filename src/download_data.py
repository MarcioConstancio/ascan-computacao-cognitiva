import urllib
import zipfile
import os

url = "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/car-plate-detection" # Endereço dos arquivos
nome_arquivo = "car-plate-detection.zip" # Nome do arquivo .zip
pasta_destino = './car-plate-detection' #Pasta para onde serão extraídos os arquivos


# Download do dataset em .zip
print('Download do conjunto de dados em andamento...')
urllib.request.urlretrieve(url, nome_arquivo)
print("Download concluído!")

# Extraindo o arquivo zipado na pasta_destino
with zipfile.ZipFile(nome_arquivo, 'r') as zip_ref:
  zip_ref.extractall(pasta_destino)

print(f'Conjunto de dados extraído para {pasta_destino}')

# Deletando o arquivo .zip
os.remove(nome_arquivo)
print(f'O arquivo {nome_arquivo} foi excluído com sucesso!')