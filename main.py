import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

app = FastAPI(title="AnyoneDS AI Engine")

# Configuração de CORS para permitir conexões do Lovable sem bloqueios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "O motor de IA do AnyoneDS está pronto para processar dados!"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Lê o ficheiro CSV suportando delimitadores comuns
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Gerar estatísticas descritivas básicas de forma segura
        summary = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            missing = int(df[col].isnull().sum())
            unique_vals = int(df[col].nunique())
            
            stats = {
                "column": col,
                "type": col_type,
                "missing": missing,
                "unique": unique_vals
            }
            
            # Adiciona métricas estatísticas apenas se a coluna for numérica
            if np.issubdtype(df[col].dtype, np.number):
                stats["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0
                stats["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else 0.0
                stats["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else 0.0
            else:
                stats["mean"] = None
                stats["min"] = None
                stats["max"] = None
            summary.append(stats)
            
        # Obter as primeiras 10 linhas para pré-visualização no dashboard
        preview = df.head(10).replace({np.nan: None}).to_dict(orient="records")
        
        return {
            "columns": list(df.columns),
            "rows_count": len(df),
            "cols_count": len(df.columns),
            "summary": summary,
            "preview": preview
        }
    except Exception as e:
        # Retorna erro 400 em vez de 500 para fornecer feedback explícito ao frontend
        raise HTTPException(status_code=400, detail=f"Erro ao analisar o ficheiro CSV: {str(e)}")

@app.post("/train")
async def train(
    file: UploadFile = File(...),
    target_col: str = Form(...),
    problem_type: str = Form(...),
    model_name: str = Form(...),
    test_size: float = Form(...)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if target_col not in df.columns:
            raise ValueError(f"A coluna alvo '{target_col}' não existe no ficheiro carregado.")
            
        # 1. Tratamento robusto de valores em falta (NaN)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Desconhecido")
        
        # 2. Divisão entre Variável Alvo (y) e Recursos (X)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # 3. Tratamento de variáveis de texto (One-Hot Encoding automático para colunas não-numéricas)
        X = pd.get_dummies(X, drop_first=True)
        
        # Converte booleanos de colunas dummy em inteiros (0 e 1) para evitar incompatibilidade
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
                
        # Garante que temos dados suficientes para realizar o treino
        if len(X) < 3:
            raise ValueError("O conjunto de dados contém poucas linhas para realizar um treino consistente.")

        # 4. Divisão em conjuntos de Treino e Teste de forma dinâmica
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # 5. Inicialização dinâmica do algoritmo selecionado pelo utilizador
        model = None
        if problem_type == "regression":
            if model_name == "linear_regression":
                model = LinearRegression()
            elif model_name == "decision_tree":
                model = DecisionTreeRegressor(random_state=42)
            elif model_name == "knn":
                model = KNeighborsRegressor()
            else:
                raise ValueError(f"Algoritmo de regressão '{model_name}' não é suportado.")
                
        elif problem_type == "classification":
            # Para classificação, garantimos que o target é interpretado de forma discreta
            y_train = y_train.astype(str)
            y_test = y_test.astype(str)
            
            if model_name == "logistic_regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "decision_tree":
                model = DecisionTreeClassifier(random_state=42)
            elif model_name == "knn":
                model = KNeighborsClassifier()
            else:
                raise ValueError(f"Algoritmo de classificação '{model_name}' não é suportado.")
        else:
            raise ValueError(f"Tipo de problema '{problem_type}' não reconhecido.")
            
        # 6. Ajustar o Modelo
        model.fit(X_train, y_train)
        
        # 7. Executar as Previsões
        y_pred = model.predict(X_test)
        
        # 8. Cálculo de Métricas de Performance
        metrics = {}
        if problem_type == "regression":
            metrics["r2"] = float(r2_score(y_test, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        else:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            metrics["f1"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            
        # 9. Construção dos dados reais vs. previstos para renderização gráfica no Lovable
        real_values = [float(val) if np.issubdtype(type(val), np.number) else val for val in y_test]
        pred_values = [float(val) if np.issubdtype(type(val), np.number) else val for val in y_pred]
        
        chart_data = []
        for i in range(len(real_values)):
            chart_data.append({
                "index": i + 1,
                "real": real_values[i],
                "previsto": pred_values[i]
            })
            
        return {
            "success": True,
            "metrics": metrics,
            "chart_data": chart_data
        }
        
    except Exception as e:
        # Se ocorrer algum erro matemático ou lógico, o FastAPI informará o erro de forma limpa
        raise HTTPException(status_code=400, detail=f"Erro durante o treino do modelo: {str(e)}")
