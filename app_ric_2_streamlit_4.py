import streamlit as st
import pandas as pd
from ultralytics import YOLO
import os
import shutil
import re
import plotly.express as px
from PIL import Image
import io
import zipfile
import requests

# ... (funções download_file_from_google_drive e get_confirm_token permanecem as mesmas)
def download_file_from_google_drive(file_id, destination):
    """
    Baixa um arquivo do Google Drive a partir do seu ID.
    """
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
                
def get_confirm_token(response):
    """
    Extrai o token de confirmação de download de arquivos grandes.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# IDs dos modelos no Google Drive
MODEL_F1_ID = "10Hh3ovvDBurmD8wZYG7uRpZklMhPHo1u"
MODEL_F2_ID = "1It73Ji3ivybC2p-8b0Lr6BIAXdn_5eyf"

path_modelo_f1 = "fase_1.pt"
path_modelo_f2 = "fase_2.pt"

# --- NOVO: Função para encontrar o diretório das imagens ---
def find_image_directory(base_dir):
    """
    Procura por um diretório que contenha arquivos de imagem em uma árvore de diretórios.
    Retorna o caminho do diretório encontrado ou None.
    """
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                return root
    return None

# --- Funções principais ---
def run_yolo_predictions(path_modelo_f1, path_modelo_f2, src_dir, path_res, pasta_inferencia, arq_inferencia):
    """
    Executa as predições YOLO para as duas fases a partir de um diretório de origem.
    """
    with st.spinner('Executando a inferência YOLO...'):
        try:
            # Novo: Encontra o diretório real das imagens
            source_directory = find_image_directory(src_dir)
            if not source_directory:
                return "Erro: Nenhuma imagem encontrada no arquivo .zip. Por favor, verifique se as imagens estão em um formato suportado."

            os.makedirs(os.path.join(path_res, pasta_inferencia), exist_ok=True)
            os.makedirs(os.path.join(path_res, arq_inferencia), exist_ok=True)

            model_f1 = YOLO(path_modelo_f1)
            model_f1.predict(source=source_directory, save=True, save_crop=True, project=path_res, name=pasta_inferencia, exist_ok=True)
            
            caminho_crops = os.path.join(path_res, pasta_inferencia, 'crops', 'Trilho')
            
            if not os.path.exists(caminho_crops) or not os.listdir(caminho_crops):
                return "Aviso: Nenhuma detecção de trilho na Fase 1. A pasta de crops está vazia. Não é possível executar a Fase 2."

            model_f2 = YOLO(path_modelo_f2)
            model_f2.predict(source=caminho_crops, save=True, save_crop=True, project=path_res, name=arq_inferencia, exist_ok=True)

            return "Inferência YOLO concluída com sucesso para ambas as fases."
        except Exception as e:
            return f"Erro durante a inferência YOLO: {e}"

def processar_arquivos(diretorio_principal):
    """... (função permanece a mesma) ..."""
    # ... (código da função processar_arquivos) ...

# ... (Layout e lógica do Streamlit, que também permanece o mesmo, exceto pelo uso da nova função na chamada de run_yolo_predictions)
st.title("Análise de RCF - Imagens RIV")
st.markdown("---")
st.header("Upload dos Dados")

uploaded_zip_file = st.file_uploader("Carregue as imagens em um arquivo .zip", type=["zip"])

st.markdown("---")

if st.button('Executar Análise', type='primary'):
    if not uploaded_zip_file:
        st.error("Por favor, carregue o arquivo .zip com as imagens para a análise.")
    else:
        st.subheader("Status da Execução")
        
        temp_dir = "temp_data"
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        try:
            with st.spinner("Baixando os modelos..."):
                download_file_from_google_drive(MODEL_F1_ID, path_modelo_f1)
                download_file_from_google_drive(MODEL_F2_ID, path_modelo_f2)
            st.info("Modelos baixados com sucesso.")

            src_dir = os.path.join(temp_dir, "uploaded_images")
            os.makedirs(src_dir)
            with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
                zip_ref.extractall(src_dir)

            st.info("Arquivos de imagens carregados e descompactados com sucesso. Iniciando a análise...")
            
            path_res = os.path.join(temp_dir, "resultado")
            yolo_status = run_yolo_predictions(path_modelo_f1, path_modelo_f2, src_dir, path_res, 'inferencia', 'resultado_final')
            st.info(yolo_status)
            
            if "Erro" not in yolo_status and "Aviso" not in yolo_status:
                path_res_modelo = os.path.join(path_res, 'resultado_final', 'crops')
                
                if os.path.exists(path_res_modelo):
                    df, avisos_processamento = processar_arquivos(path_res_modelo)
                    
                    if avisos_processamento:
                        st.warning("Houve avisos durante o processamento de arquivos:")
                        for aviso in avisos_processamento:
                            st.text(f"- {aviso}")

                    if not df.empty:
                        st.success("Processamento de arquivos concluído e DataFrame gerado.")
                        
                        st.subheader("Prévia do DataFrame")
                        st.dataframe(df)

                        st.subheader("Download dos Relatórios")
                        
                        csv_buffer = io.StringIO()
                        df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Baixar Relatório CSV",
                            data=csv_buffer.getvalue(),
                            file_name='relatorio.csv',
                            mime='text/csv',
                        )

                        xlsx_buffer = io.BytesIO()
                        df.to_excel(xlsx_buffer, index=False)
                        st.download_button(
                            label="📥 Baixar Relatório XLSX",
                            data=xlsx_buffer.getvalue(),
                            file_name='relatorio.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        )
                        
                        st.subheader("Análises Visuais (Plotly)")

                        st.markdown("### Contagem de Classificações por Pátio")
                        if 'Classificação' in df.columns and 'Pátio' in df.columns:
                            classificacao_por_patio = df.groupby(['Pátio', 'Classificação']).size().reset_index(name='Contagem')
                            fig_bar = px.bar(classificacao_por_patio, x='Pátio', y='Contagem', color='Classificação', 
                                             title='Contagem de Defeitos por Pátio')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("Dados para a visualização 'Classificação por Pátio' não estão disponíveis no DataFrame.")
                        
                        st.markdown("### Distribuição de Defeitos ao Longo dos KMs")
                        if 'KM' in df.columns and 'Classificação' in df.columns:
                            fig_scatter = px.scatter(df, x='KM', y='Metro', color='Classificação', 
                                                     title='Localização de Defeitos por KM e Metro')
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("Dados para a visualização 'Distribuição de Defeitos' não estão disponíveis no DataFrame.")
                    else:
                        st.warning("O DataFrame está vazio. Nenhum arquivo processado ou com dados válidos.")
                else:
                    st.error("O diretório de resultados da Fase 2 não foi encontrado.")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(path_modelo_f1):
                os.remove(path_modelo_f1)
            if os.path.exists(path_modelo_f2):
                os.remove(path_modelo_f2)
            st.info("Arquivos temporários limpos.")
            