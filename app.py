import streamlit as st
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import pipeline  # Importação para o modelo LLM

# ---------------- O carregamento para FASTAPI Inicialização -------------------
app_fastapi = FastAPI()

# ------------- Aqui está o Carregamento do Modelo LLM -----------------
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except Exception as e:
    sentiment_analyzer = None
    print(f"Erro ao carregar o modelo: {e}")

# --------------Modelos de Dados para a API -------------------
class Projeto(BaseModel):
    id: int
    nome: str
    status: str

# Modelo de dados para entrada de texto
class TextInput(BaseModel):
    text: str

# Modelo de dados para saída de análise de sentimento
class SentimentOutput(BaseModel):
    text: str
    label: str
    score: float

# ---------------- Dados Temporários -------------------
# Dados temporários armazenados em memória
data = [
    {"id": 1, "nome": "Projeto Social 1", "status": "Ativo"},
    {"id": 2, "nome": "Projeto Social 2", "status": "Concluído"}
]

# ---------------- FASTAPI Endpoints -------------------
# Rota GET para consultar todos os projetos
@app_fastapi.get("/projetos", response_model=List[Projeto])
def get_projetos():
    return data

# Rota GET para consultar um projeto específico pelo ID
@app_fastapi.get("/projetos/{projeto_id}", response_model=Projeto)
def get_projeto(projeto_id: int):
    projeto = next((item for item in data if item["id"] == projeto_id), None)
    if projeto is None:
        raise HTTPException(status_code=404, detail="Projeto não encontrado")
    return projeto

# Rota POST para adicionar um novo projeto
@app_fastapi.post("/projetos", response_model=Projeto)
def add_projeto(projeto: Projeto):
    if any(item["id"] == projeto.id for item in data):
        raise HTTPException(status_code=400, detail="ID já existe")
    new_project = projeto.dict()
    data.append(new_project)
    return new_project

# Rota POST para processar texto e retornar análise de sentimento
@app_fastapi.post("/processar_texto", response_model=SentimentOutput)
async def processar_texto(input_data: TextInput):
    """
    Processa o texto fornecido e retorna a análise de sentimento.
    """
    if sentiment_analyzer is None:
        raise HTTPException(status_code=503, detail="Modelo de linguagem não disponível.")
    try:
        result = sentiment_analyzer(input_data.text)[0]
        return SentimentOutput(
            text=input_data.text,
            label=result['label'],
            score=result['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o texto: {str(e)}")

# ---------------- STREAMLIT Funcionalidades -------------------
@st.cache_data
def fetch_reliefweb_projects(query="projects", limit=10):
    url = "https://api.reliefweb.int/v1/reports"
    params = {
        "appname": "apidoc",
        "query[value]": query,
        "limit": limit,
        "profile": "full"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.error(f"Erro ao acessar a API da ReliefWeb. Status code: {response.status_code}")
        return []

# Processar dados da API em um DataFrame
def process_project_data(projects):
    project_list = []
    for project in projects:
        title = project['fields'].get('title', 'N/A')
        country = project['fields'].get('primary_country', {}).get('name', 'N/A')
        date = project['fields'].get('date', {}).get('created', 'N/A')
        summary = project['fields'].get('body', 'N/A')
        url = project.get('href', 'N/A')

        project_list.append({
            "Título": title,
            "País": country,
            "Data de Criação": date,
            "Resumo": summary,
            "URL": url
        })

    return pd.DataFrame(project_list)

# Função para fazer upload de um arquivo CSV e ler o conteúdo
def upload_csv():
    uploaded_file = st.file_uploader("Faça upload de um arquivo CSV para adicionar mais informações", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

# Função para baixar um DataFrame como CSV
def download_csv(df):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Baixar dados como CSV",
        data=csv,
        file_name="dados_projetos.csv",
        mime="text/csv"
    )

# ---------------- STREAMLIT Páginas -------------------
def pagina_home():
    st.title("Bem-vindo à Aplicação de Múltiplas Páginas")
    st.write("""
    Esta aplicação permite carregar, visualizar e analisar dados de maneira interativa.
    Use o menu lateral para navegar entre as diferentes páginas.
    """)

def pagina_upload():
    st.title("Upload de Arquivos CSV")
    additional_data = upload_csv()
    if additional_data is not None:
        st.write("Dados carregados com sucesso!")
        st.dataframe(additional_data.head())
    else:
        st.write("Nenhum arquivo CSV foi carregado.")

def pagina_visualizacao():
    st.title("Visualização de Projetos da ReliefWeb")
    # Inicializar o número de projetos no estado de sessão
    if 'num_projects' not in st.session_state:
        st.session_state.num_projects = 5  # Valor inicial padrão

    # Controle do número de projetos exibidos
    num_projects = st.slider("Número de projetos", min_value=1, max_value=50, value=st.session_state.num_projects)
    st.session_state.num_projects = num_projects  # Atualizar o estado

    # Carregar dados da API
    projects = fetch_reliefweb_projects(limit=st.session_state.num_projects)

    if projects:
        project_df = process_project_data(projects)
        st.write("Projetos Carregados:")
        st.dataframe(project_df.head())

        # Opção para download dos dados
        download_csv(project_df)
    else:
        st.write("Nenhum projeto encontrado.")

def pagina_estatisticas():
    st.title("Análise Estatística de Dados de Projetos")

    # Exemplo de dados fictícios para análise estatística
    data = pd.DataFrame({
        'Categoria': ['A', 'B', 'C', 'D'],
        'Valores': [23, 45, 56, 78]
    })

    st.write("Estatísticas descritivas dos dados:")
    st.write(data.describe())

    st.bar_chart(data['Valores'])

def pagina_analise_sentimento():
    st.title("Análise de Sentimento de Texto")
    st.write("Digite um texto para receber a análise de sentimento.")

    texto_usuario = st.text_area("Texto:", "")
    if st.button("Analisar Sentimento"):
        if texto_usuario.strip():
            try:
                resposta = requests.post(
                    "http://127.0.0.1:8000/processar_texto",
                    json={"text": texto_usuario}
                )
                if resposta.status_code == 200:
                    resultado = resposta.json()
                    st.subheader("Resultado da Análise:")
                    st.write(f"**Texto:** {resultado['text']}")
                    st.write(f"**Sentimento:** {resultado['label']}")
                    st.write(f"**Confiança:** {resultado['score']:.2f}")
                else:
                    erro = resposta.json().get('detail', 'Erro desconhecido.')
                    st.error(f"Erro na API: {erro}")
            except Exception as e:
                st.error(f"Erro ao conectar com a API: {e}")
        else:
            st.warning("Por favor, insira um texto para análise.")

# ---------------- STREAMLIT Navegação -------------------
# Menu de navegação lateral
st.sidebar.title("Menu de Navegação")
menu_options = ["Home", "Upload de Dados", "Visualização de Projetos", "Estatísticas", "Análise de Sentimento"]
choice = st.sidebar.selectbox("Selecione a Página:", menu_options)

# Navegação entre as páginas
if choice == "Home":
    pagina_home()
elif choice == "Upload de Dados":
    pagina_upload()
elif choice == "Visualização de Projetos":
    pagina_visualizacao()
elif choice == "Estatísticas":
    pagina_estatisticas()
elif choice == "Análise de Sentimento":
    pagina_analise_sentimento()
