from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import json
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

app = FastAPI(title="AnyoneDS - Motor de Processamento Científico")

# Habilitar CORS para o Lovable poder chamar a API de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "message": "AnyoneDS AI Engine is running!"}

@app.post("/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Recebe a planilha e faz uma análise estatística preliminar"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Estatísticas básicas
        total_rows = int(df.shape[0])
        total_cols = int(df.shape[1])
        missing_values = int(df.isnull().sum().sum())
        
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        categorical_cols = [col for col in df.columns if col not in numeric_cols]
        
        # Gerar sumário descritivo simples das colunas
        summary = {}
        for col in numeric_cols:
            summary[col] = {
                "mean": float(df[col].mean()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "std": float(df[col].std())
            }

        return {
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "shape": {"rows": total_rows, "cols": total_cols},
            "missing_values": missing_values,
            "summary_statistics": summary,
            "preview_data": json.loads(df.head(10).to_json(orient="records"))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo: {str(e)}")

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    problem_type: str = Form(...), # "regressao" ou "classificacao"
    model_name: str = Form(...),   # "linear_regression", "decision_tree", "knn"
    test_size: float = Form(...)   # Ex: 0.2
):
    """Treina o modelo baseado no input do usuário no Lovable e retorna resultados"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if target_col not in df.columns:
            raise HTTPException(status_code=400, detail="Coluna alvo não encontrada na planilha")
            
        # Remover valores nulos básicos para evitar erros de ML
        df = df.dropna()
        
        # Separar variáveis X e y
        # Para este MVP simples, vamos focar apenas em colunas numéricas como preditores
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != target_col]
        
        if len(numeric_cols) == 0:
            raise HTTPException(status_code=400, detail="Não há variáveis numéricas suficientes para o treino.")
            
        X = df[numeric_cols]
        y = df[target_col]
        
        # Split de dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Selecionar e instanciar o modelo correto
        model = None
        if problem_type == "regressao":
            if model_name == "linear_regression":
                model = LinearRegression()
            elif model_name == "decision_tree":
                model = DecisionTreeRegressor(max_depth=5)
            elif model_name == "knn":
                model = KNeighborsRegressor(n_neighbors=5)
        elif problem_type == "classificacao":
            if model_name == "linear_regression": # Na vdd logística para classificação
                model = LogisticRegression()
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(max_depth=5)
            elif model_name == "knn":
                model = KNeighborsClassifier(n_neighbors=5)
                
        if model is None:
            raise HTTPException(status_code=400, detail="Modelo inválido selecionado")
            
        # Treinar o algoritmo
        model.fit(X_train, y_train)
        
        # Realizar previsões na base de teste
        y_pred = model.predict(X_test)
        
        # Calcular as métricas científicas
        metrics = {}
        if problem_type == "regressao":
            metrics["r2"] = float(r2_score(y_test, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        else:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["precision"] = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # Estruturar dados de saída para os gráficos no Lovable (Real vs Previsto)
        chart_data = []
        for real, pred in zip(y_test.values, y_pred):
            chart_data.append({
                "real": float(real),
                "predicted": float(pred)
            })
            
        return {
            "metrics": metrics,
            "chart_data": chart_data[:100], # Limitar a 100 pontos para manter a renderização leve no React
            "features_used": numeric_cols
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no treino: {str(e)}")
