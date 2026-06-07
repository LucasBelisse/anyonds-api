import io
import pandas as pd
import numpy as np
import sqlite3
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

app = FastAPI(title="AnyoneDS AI & SQL Engine")

# Configuração de CORS para permitir conexões do Lovable sem bloqueios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_nan_values(data):
    """
    Função recursiva e blindada para remover NaNs, Infinities e tipos NumPy incompatíveis,
    convertendo-os de forma segura em 'None' (null no JSON) ou tipos primitivos do Python.
    """
    if isinstance(data, list):
        return [clean_nan_values(item) for item in data]
    elif isinstance(data, dict):
        return {k: clean_nan_values(v) for k, v in data.items()}
    elif isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data) or pd.isna(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif pd.isna(data):
        return None
    else:
        return data

def safe_read_csv(contents: bytes) -> pd.DataFrame:
    """
    Decodifica bytes do arquivo e detecta automaticamente o delimitador de colunas (vírgula ou ponto e vírgula),
    comportando-se de forma flexível para arquivos salvos em Excel com localidade brasileira.
    """
    decoded = None
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
        try:
            decoded = contents.decode(encoding)
            break
        except Exception:
            continue
            
    if decoded is None:
        decoded = contents.decode('utf-8', errors='ignore')
        
    try:
        # Tenta inferir o delimitador olhando a primeira linha
        first_line = decoded.split('\n')[0] if '\n' in decoded else decoded
        if ';' in first_line and ',' not in first_line:
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = ','
            
        df = pd.read_csv(io.StringIO(decoded), sep=sep)
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(decoded), sep=None, engine='python')
        except Exception:
            df = pd.read_csv(io.StringIO(decoded))
            
    # Limpa espaços em branco indesejados nas colunas
    df.columns = [str(c).strip() for c in df.columns]
    return df

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "O motor de IA e SQL do AnyoneDS está pronto para processar dados!"
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = safe_read_csv(contents)
        
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
            
            # Tratamento blindado de métricas estatísticas de nulos e infinitos
            if np.issubdtype(df[col].dtype, np.number):
                col_clean = df[col].dropna()
                
                # CORREÇÃO CIRÚRGICA: Só testa infinitos (isinf) se a coluna for do tipo decimal (float/floating)
                # Números inteiros (int64) geravam erro de tipo nesta função e causavam o crash do HTTP 400
                if np.issubdtype(df[col].dtype, np.floating):
                    col_clean = col_clean[~np.isinf(col_clean)]
                
                if len(col_clean) > 0:
                    stats["mean"] = clean_nan_values(col_clean.mean())
                    stats["min"] = clean_nan_values(col_clean.min())
                    stats["max"] = clean_nan_values(col_clean.max())
                else:
                    stats["mean"] = None
                    stats["min"] = None
                    stats["max"] = None
            else:
                stats["mean"] = None
                stats["min"] = None
                stats["max"] = None
            summary.append(stats)
            
        # Obter prévia das primeiras 15 linhas limpas de NaNs de forma recursiva
        raw_preview = df.head(15).to_dict(orient="records")
        preview = clean_nan_values(raw_preview)
        
        return clean_nan_values({
            "columns": list(df.columns),
            "rows_count": len(df),
            "cols_count": len(df.columns),
            "summary": summary,
            "preview": preview
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao analisar o ficheiro CSV: {str(e)}")

@app.post("/sql-query")
async def sql_query(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Carrega o arquivo CSV dinamicamente em uma tabela SQLite em memória
    e executa a query SQL do usuário em tempo real!
    """
    try:
        contents = await file.read()
        df = safe_read_csv(contents)
        
        # Cria uma conexão com banco SQLite em memória
        conn = sqlite3.connect(":memory:")
        
        # Salva o DataFrame como uma tabela SQL chamada 'dataset'
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        
        # Executa a query fornecida pelo usuário
        result_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Sanitização robusta dos resultados da query SQL
        raw_results = result_df.head(100).to_dict(orient="records")
        result_preview = clean_nan_values(raw_results)
        
        return clean_nan_values({
            "success": True,
            "columns": list(result_df.columns),
            "rows_count": len(result_df),
            "preview": result_preview,
            "query_executed": query
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro de sintaxe SQL: {str(e)}")

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
        df = safe_read_csv(contents)
        
        if target_col not in df.columns:
            raise ValueError(f"A coluna alvo '{target_col}' não existe no ficheiro carregado.")
            
        # Limpeza robusta antes de rodar o pipeline de ML
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Desconhecido")
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Conversão de colunas categóricas para dummies
        X = pd.get_dummies(X, drop_first=True)
        
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
                
        if len(X) < 3:
            raise ValueError("O conjunto de dados contém poucas linhas para realizar um treino consistente.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
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
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
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
            
        real_values = [float(val) if np.issubdtype(type(val), np.number) else val for val in y_test]
        pred_values = [float(val) if np.issubdtype(type(val), np.number) else val for val in y_pred]
        
        chart_data = []
        for i in range(len(real_values)):
            chart_data.append({
                "index": i + 1,
                "real": real_values[i],
                "previsto": pred_values[i]
            })
            
        return clean_nan_values({
            "success": True,
            "metrics": metrics,
            "chart_data": chart_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante o treino do modelo: {str(e)}")
