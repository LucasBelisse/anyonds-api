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
            
            # Tratamento blindado de métricas estatísticas de nulos e infinitos
            if np.issubdtype(df[col].dtype, np.number):
                mean_val = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                
                stats["mean"] = float(mean_val) if pd.notnull(mean_val) and not np.isinf(mean_val) else None
                stats["min"] = float(min_val) if pd.notnull(min_val) and not np.isinf(min_val) else None
                stats["max"] = float(max_val) if pd.notnull(max_val) and not np.isinf(max_val) else None
            else:
                stats["mean"] = None
                stats["min"] = None
                stats["max"] = None
            summary.append(stats)
            
        # BLINDAGEM JSON: Obtém prévia das 15 linhas limpando NaN/inf por None recursivamente
        raw_preview = df.head(15).to_dict(orient="records")
        preview = []
        for record in raw_preview:
            cleaned_record = {}
            for k, v in record.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    cleaned_record[k] = None
                else:
                    cleaned_record[k] = v
            preview.append(cleaned_record)
        
        return {
            "columns": list(df.columns),
            "rows_count": len(df),
            "cols_count": len(df.columns),
            "summary": summary,
            "preview": preview
        }
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
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Cria uma conexão com banco SQLite em memória
        conn = sqlite3.connect(":memory:")
        
        # Salva o DataFrame como uma tabela SQL chamada 'dataset'
        df.to_sql("dataset", conn, index=False, if_exists="replace")
        
        # Executa a query fornecida pelo usuário
        result_df = pd.read_sql_query(query, conn)
        conn.close()
        
        # BLINDAGEM JSON: Substitui valores NaN por None de forma limpa para o SQLite
        raw_results = result_df.head(100).to_dict(orient="records")
        result_preview = []
        for record in raw_results:
            cleaned_record = {}
            for k, v in record.items():
                if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                    cleaned_record[k] = None
                else:
                    cleaned_record[k] = v
            result_preview.append(cleaned_record)
        
        return {
            "success": True,
            "columns": list(result_df.columns),
            "rows_count": len(result_df),
            "preview": result_preview,
            "query_executed": query
        }
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
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if target_col not in df.columns:
            raise ValueError(f"A coluna alvo '{target_col}' não existe no ficheiro carregado.")
            
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Desconhecido")
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
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
            
        return {
            "success": True,
            "metrics": metrics,
            "chart_data": chart_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro durante o treino do modelo: {str(e)}")
