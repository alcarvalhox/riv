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
import gdown

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Análise RCF - Imagens RIV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LAYOUT E ESTILO ---
# Injetando CSS para o título e o cabeçalho
st.markdown("""
<style>
/* Estilo para a área do título principal */
.main-title-container {
    background-color: yellow;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: black;
    margin-bottom: 20px;
}
/* Estilo para o cabeçalho "Upload dos Dados" */
.upload-header {
    background-color: darkblue;
    padding: 10px;
    border-radius: 5px;
    color: white;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}
/* Estilo para os cards de resultado */
.result-card-odd {
    background-color: yellow;
    color: black;
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
}
.result-card-even {
    background-color: darkblue;
    color: white;
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Layout com colunas para as logos
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    st.image("logo_mrs.jpg", width=100)
with col2:
    st.markdown("<h1 class='main-title-container'>Análise de RCF - Imagens RIV</h1>", unsafe_allow_html=True)
with col3:
    st.image("riv.jpg", width=100)

# --- IDs dos modelos no Google Drive ---
MODEL_F1_ID = "10Hh3ovvDBurmD8wZYG7uRpZklMhPHo1u"
MODEL_F2_ID = "1It73Ji3ivybC2p-8b0Lr6BIAXdn_5eyf"

path_modelo_f1 = "fase_1.pt"
path_modelo_f2 = "fase_2.pt"

# --- Funções auxiliares ---
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

def run_yolo_predictions(path_modelo_f1, path_modelo_f2, src_dir, path_res, pasta_inferencia, arq_inferencia):
    """
    Executa as predições YOLO para as duas fases a partir de um diretório de origem.
    """
    with st.spinner('Executando a inferência YOLO...'):
        try:
            source_directory = find_image_directory(src_dir)
            if not source_directory:
                return "Erro: Nenhuma imagem encontrada no arquivo .zip. Por favor, verifique se as imagens estão em um formato suportado e se o arquivo .zip não está vazio."

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

# --- NOVO FLUXO DE AUTENTICAÇÃO SIMPLES ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

USERNAME = "riv"
PASSWORD = "123"

if not st.session_state['authenticated']:
    st.sidebar.title("Login")
    username_input = st.sidebar.text_input("Usuário")
    password_input = st.sidebar.text_input("Senha", type="password")

    if st.sidebar.button("Entrar"):
        if username_input == USERNAME and password_input == PASSWORD:
            st.session_state['authenticated'] = True
            st.success("Login realizado com sucesso!")
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

if st.session_state['authenticated']:
    st.sidebar.button("Sair", on_click=lambda: st.session_state.update(authenticated=False))

    # Título do aplicativo já foi definido com CSS
    st.markdown("<h2 class='upload-header'>Upload dos Dados</h2>", unsafe_allow_html=True)

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
                with st.spinner("Baixando os modelos do Google Drive..."):
                    gdown.download(id=MODEL_F1_ID, output=path_modelo_f1, quiet=True)
                    gdown.download(id=MODEL_F2_ID, output=path_modelo_f2, quiet=True)
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
                                
                            st.subheader("Resultados Detalhados (Alternando Cores)")
                            for index, row in df.iterrows():
                                style_class = 'result-card-odd' if index % 2 == 0 else 'result-card-even'
                                st.markdown(
                                    f"""
                                    <div class='{style_class}'>
                                        <strong>Linha:</strong> {row['Linha']}<br>
                                        <strong>Pátio:</strong> {row['Pátio']}<br>
                                        <strong>Data:</strong> {row['Dia']}/{row['Mês']}/{row['Ano']}<br>
                                        <strong>KM:</strong> {row['KM']}, <strong>Metro:</strong> {row['Metro']}<br>
                                        <strong>Classificação:</strong> {row['Classificação']}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
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
                
                
    
    
            
