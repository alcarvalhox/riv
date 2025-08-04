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

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Análise RCF - Imagens RIV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções auxiliares para download de arquivos do Google Drive ---
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
        # Itera sobre o conteúdo do arquivo para baixá-lo em pedaços
        for chunk in response.iter_content(32768):
            if chunk:  # Filtra pacotes vazios
                f.write(chunk)
                
def get_confirm_token(response):
    """
    Extrai o token de confirmação de download de arquivos grandes.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# --- IDs dos modelos no Google Drive ---
# Extraídos dos links fornecidos
MODEL_F1_ID = "10Hh3ovvDBurmD8wZYG7uRpZklMhPHo1u"
MODEL_F2_ID = "1It73Ji3ivybC2p-8b0Lr6BIAXdn_5eyf"

# Nomes locais dos arquivos
path_modelo_f1 = "fase_1.pt"
path_modelo_f2 = "fase_2.pt"

# --- Funções do seu código original (adaptadas) ---
def run_yolo_predictions(path_modelo_f1, path_modelo_f2, src_dir, path_res, pasta_inferencia, arq_inferencia):
    """
    Executa as predições YOLO para as duas fases a partir de um diretório de origem.
    """
    with st.spinner('Executando a inferência YOLO...'):
        try:
            os.makedirs(os.path.join(path_res, pasta_inferencia), exist_ok=True)
            os.makedirs(os.path.join(path_res, arq_inferencia), exist_ok=True)

            model_f1 = YOLO(path_modelo_f1)
            model_f1.predict(source=src_dir, save=True, save_crop=True, project=path_res, name=pasta_inferencia, exist_ok=True)
            
            caminho_crops = os.path.join(path_res, pasta_inferencia, 'crops', 'Trilho')
            
            if not os.path.exists(caminho_crops) or not os.listdir(caminho_crops):
                return "Aviso: Nenhuma detecção de trilho na Fase 1. A pasta de crops está vazia. Não é possível executar a Fase 2."

            model_f2 = YOLO(path_modelo_f2)
            model_f2.predict(source=caminho_crops, save=True, save_crop=True, project=path_res, name=arq_inferencia, exist_ok=True)

            return "Inferência YOLO concluída com sucesso para ambas as fases."
        except Exception as e:
            return f"Erro durante a inferência YOLO: {e}"

def processar_arquivos(diretorio_principal):
    """
    Processa os arquivos em um diretório e seus subdiretórios,
    extraindo as informações do nome e criando um DataFrame.
    """
    dados = []
    avisos = []
    for root, dirs, files in os.walk(diretorio_principal):
        for file in files:
            match = re.match(
                r"^(?P<lim_sup>\d+)\s+-\s+(?P<lim_inf>\d+)\s*(?P<linha>[A-Z\d]+)_(?P<patio>[A-Za-z]+)_(?P<data>\d{8})_(?P<km>\d+)_(?P<metro>\d+)\.jpg$",
                file
            )
            
            if not match:
                avisos.append(f"Aviso: O arquivo '{file}' não segue o padrão esperado e foi ignorado.")
                continue

            try:
                lim_sup = int(match.group('lim_sup'))
                lim_inf = int(match.group('lim_inf'))
                linha = match.group('linha')
                patio = match.group('patio')
                data_str = match.group('data')
                km = int(match.group('km'))
                metro = int(match.group('metro'))
                
                data_obj = pd.to_datetime(data_str, format='%Y%m%d')
                
                dados.append({
                    'LIM_sup': lim_sup,
                    'LIM_inf': lim_inf,
                    'Linha': linha,
                    'Pátio': patio,
                    'Ano': data_obj.year,
                    'Mês': data_obj.month,
                    'Dia': data_obj.day,
                    'KM': km,
                    'Metro': metro,
                    'Classificação': os.path.basename(root)
                })
            except (IndexError, AttributeError, ValueError) as e:
                avisos.append(f"Erro ao processar arquivo '{file}': {e}. Foi ignorado.")
                continue

    df = pd.DataFrame(dados)
    return df, avisos

# --- Layout e Lógica do Aplicativo Streamlit ---
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
            # Baixa os modelos do Google Drive
            with st.spinner("Baixando os modelos..."):
                download_file_from_google_drive(MODEL_F1_ID, path_modelo_f1)
                download_file_from_google_drive(MODEL_F2_ID, path_modelo_f2)
            st.info("Modelos baixados com sucesso.")

            # Descompacta o arquivo .zip de imagens
            src_dir = os.path.join(temp_dir, "uploaded_images")
            os.makedirs(src_dir)
            with zipfile.ZipFile(uploaded_zip_file, 'r') as zip_ref:
                zip_ref.extractall(src_dir)

            st.info("Arquivos de imagens carregados e descompactados com sucesso. Iniciando a análise...")
            
            # Executa a inferência YOLO
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
            # Limpa todos os arquivos temporários, incluindo os modelos baixados
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if os.path.exists(path_modelo_f1):
                os.remove(path_modelo_f1)
            if os.path.exists(path_modelo_f2):
                os.remove(path_modelo_f2)
            st.info("Arquivos temporários limpos.")
            