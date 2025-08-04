import streamlit as st
import pandas as pd
from ultralytics import YOLO
import os
import shutil
import re
import plotly.express as px
from PIL import Image

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Análise RCF - Imagens RIV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funções do seu código original (adaptadas) ---

def run_yolo_predictions(path_modelo_f1, path_modelo_f2, src, path_res, pasta_inferencia, arq_inferencia):
    """
    Executa as predições YOLO para as duas fases e retorna o status.
    """
    with st.spinner('Executando a inferência YOLO...'):
        try:
            # Limpa as pastas de resultados
            if os.path.exists(os.path.join(path_res, pasta_inferencia)):
                shutil.rmtree(os.path.join(path_res, pasta_inferencia))
            if os.path.exists(os.path.join(path_res, arq_inferencia)):
                shutil.rmtree(os.path.join(path_res, arq_inferencia))

            os.makedirs(os.path.join(path_res, pasta_inferencia), exist_ok=True)
            os.makedirs(os.path.join(path_res, arq_inferencia), exist_ok=True)

            # Fase 1 da inferência
            model_f1 = YOLO(path_modelo_f1)
            model_f1.predict(source=src, save=True, save_crop=True, project=path_res, name=pasta_inferencia, exist_ok=True)
            
            caminho_crops = os.path.join(path_res, pasta_inferencia, 'crops', 'Trilho')
            
            if not os.path.exists(caminho_crops) or not os.listdir(caminho_crops):
                return "Aviso: Nenhuma detecção de trilho na Fase 1. A pasta de crops está vazia. Não é possível executar a Fase 2."

            # Fase 2 da inferência
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
                
                # A coluna 'Data' foi removida conforme a solicitação
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

def clean_source_directory(path):
    """
    Deleta todos os arquivos e subdiretórios de um diretório.
    """
    if not os.path.exists(path):
        return f"Aviso: O diretório {path} não existe."
    
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            return f"Erro ao deletar {item_path}: {e}"
            
    return f"Conteúdo do diretório '{path}' deletado com sucesso."

# --- Layout e Lógica do Aplicativo Streamlit ---

st.title("Análise de RCF - Imagens RIV")

# Sidebar para configurações de caminho
st.sidebar.header("Configurações de Caminho")

# Usando st.text_input para permitir que o usuário digite ou cole o caminho
# O valor inicial está vazio para forçar o usuário a preencher
path_modelo_f1 = st.sidebar.text_input("Caminho do Modelo Fase 1 (.pt)", "")
path_modelo_f2 = st.sidebar.text_input("Caminho do Modelo Fase 2 (.pt)", "")
src = st.sidebar.text_input("Caminho dos Dados Originais (Pasta)", "")
path_rel = st.sidebar.text_input("Caminho para Salvar Relatórios (Pasta)", "")
path_res = st.sidebar.text_input("Caminho para Salvar Resultados (Pasta)", "")

st.sidebar.markdown("---")

# Botão para validar e executar a análise
if st.sidebar.button('Executar Análise'):
    
    # 1. Validação inicial dos caminhos
    if not all([path_modelo_f1, path_modelo_f2, src, path_rel, path_res]):
        st.error("Por favor, preencha todos os campos de caminho na barra lateral.")
    else:
        st.subheader("Status da Execução")
        
        # 2. Verificação se os caminhos existem
        caminhos_para_verificar = {
            "Modelo Fase 1": path_modelo_f1,
            "Modelo Fase 2": path_modelo_f2,
            "Dados Originais": src,
        }
        
        erros_verificacao = []
        for nome, caminho in caminhos_para_verificar.items():
            if not os.path.exists(caminho):
                erros_verificacao.append(f"Erro: O caminho de {nome} '{caminho}' não existe.")

        if erros_verificacao:
            for erro in erros_verificacao:
                st.error(erro)
        else:
            # 3. Executa a inferência YOLO
            yolo_status = run_yolo_predictions(path_modelo_f1, path_modelo_f2, src, path_res, 'inferencia', 'resultado_final')
            st.info(yolo_status)
            
            if "Erro" not in yolo_status and "Aviso" not in yolo_status:
                
                # 4. Processa os arquivos resultantes
                path_res_modelo = os.path.join(path_res, 'resultado_final', 'crops')
                
                if os.path.exists(path_res_modelo):
                    df, avisos_processamento = processar_arquivos(path_res_modelo)
                    
                    if avisos_processamento:
                        st.warning("Houve avisos durante o processamento de arquivos:")
                        for aviso in avisos_processamento:
                            st.text(f"- {aviso}")

                    # 5. Salva e exibe o DataFrame
                    if not df.empty:
                        st.success("Processamento de arquivos concluído e DataFrame gerado.")
                        
                        # Garantindo que as pastas existam
                        os.makedirs(path_rel, exist_ok=True)
                        path_relatorio_csv = os.path.join(path_rel, 'relatorio.csv')
                        path_relatorio_xlsx = os.path.join(path_rel, 'relatorio.xlsx')
                        
                        df.to_csv(path_relatorio_csv, index=False)
                        df.to_excel(path_relatorio_xlsx, index=False)
                        st.info(f"Relatórios salvos em: {path_rel}")
                        
                        st.subheader("Prévia do DataFrame")
                        st.dataframe(df)

                        st.subheader("Análises Visuais (Plotly)")

                        # Criando visualizações com Plotly
                        st.markdown("### Contagem de Classificações por Pátio")
                        if 'Classificação' in df.columns and 'Pátio' in df.columns:
                            classificacao_por_patio = df.groupby(['Pátio', 'Classificação']).size().reset_index(name='Contagem')
                            fig_bar = px.bar(classificacao_por_patio, x='Pátio', y='Contagem', color='Classificação', 
                                             title='Contagem de Defeitos por Pátio',
                                             labels={'Pátio': 'Pátio', 'Contagem': 'Número de Ocorrências'})
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("Dados para a visualização 'Classificação por Pátio' não estão disponíveis no DataFrame.")
                        
                        st.markdown("### Distribuição de Defeitos ao Longo dos KMs")
                        if 'KM' in df.columns and 'Classificação' in df.columns:
                            fig_scatter = px.scatter(df, x='KM', y='Metro', color='Classificação', 
                                                     title='Localização de Defeitos por KM e Metro',
                                                     labels={'KM': 'Quilômetro', 'Metro': 'Metro'})
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("Dados para a visualização 'Distribuição de Defeitos' não estão disponíveis no DataFrame.")
                            
                    else:
                        st.warning("O DataFrame está vazio. Nenhum arquivo processado ou com dados válidos.")
                else:
                    st.error("O diretório de resultados da Fase 2 não foi encontrado.")

# Botão para deletar conteúdo da pasta 'validacao' (agora com caminho dinâmico)
if st.sidebar.button("Deletar conteúdo da pasta de dados originais"):
    if st.sidebar.warning("Tem certeza? Esta ação é irreversível."):
        if st.sidebar.button("Confirmar Deleção"):
            if not src:
                st.sidebar.error("Por favor, insira o caminho da pasta de dados originais para deletar.")
            else:
                status_limpeza = clean_source_directory(src)
                st.sidebar.info(status_limpeza)