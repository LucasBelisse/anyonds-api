import io
import sqlite3
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# Configuração do CORS para o Lovable conseguir conectar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# Banco de dados SQLite em memória compartilhado para a sessão
DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)

def clean_nan_values(obj):
    """Garante que qualquer NaN ou Inf do NumPy/Pandas vire None para o JSON não quebrar"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(x) for x in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif str(type(obj)).grid_contains("float") or str(type(obj)).grid_contains("int"):
        # Captura tipos numéricos nativos do NumPy (ex: np.float64, np.int64)
        val = float(obj) if "float" in str(type(obj)) else int(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return obj

@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        # Lê o CSV tratando strings e nulos de forma agnóstica
        df = pd.read_csv(io.BytesIO(contents))
        
        # Limpa nomes de colunas removendo espaços ou caracteres especiais invisíveis
        df.columns = [c.strip() for c in df.columns]
        
        # Salva o DataFrame no SQLite limpando a tabela anterior se existir
        df.to_sql("dados_planilha", DB_CONN, if_exists="replace", index=False)
        
        columns_info = []
        
        for col in df.columns:
            null_count = int(df[col].isna().sum())
            col_type = str(df[col].dtype)
            
            # Estrutura padrão de metadados da coluna
            info = {
                "name": col,
                "type": "Misto/Texto" if col_type == "object" else col_type,
                "null_count": null_count,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0
            }
            
            # PROTEÇÃO CIRÚRGICA: Só faz cálculos matemáticos se a coluna for estritamente numérica!
            if np.issubdtype(df[col].dtype, np.number):
                col_clean = df[col].dropna()
                
                # Se for float, remove infinitos antes da média
                if np.issubdtype(df[col].dtype, np.floating):
                    col_clean = col_clean[~np.isinf(col_clean)]
                
                if not col_clean.empty:
                    info["mean"] = float(col_clean.mean())
                    info["min"] = float(col_clean.min())
                    info["max"] = float(col_clean.max())
            
            columns_info.append(info)
            
        response_data = {
            "status": "success",
            "rows": len(df),
            "columns": columns_info,
            "preview": df.head(10).fillna("").to_dict(orient="records")
        }
        
        return JSONResponse(content=clean_nan_values(response_data))
        
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"status": "error", "detail": f"Erro ao analisar colunas: {str(e)}"}
        )

@app.post("/sql-query")
async def run_query(query: str = Form(...)):
    try:
        # Executa a query SQL direto no banco em memória
        cursor = DB_CONN.cursor()
        cursor.execute(query)
        
        # Captura os resultados e os nomes das colunas resultantes
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        # Transforma o resultado em uma lista de dicionários (padrão JSON)
        result_dict = [dict(zip(columns, row)) for row in rows]
        
        response_data = {
            "status": "success",
            "columns": columns,
            "data": result_dict
        }
        
        return JSONResponse(content=clean_nan_values(response_data))
        
    except Exception as e:
        return JSONResponse(
            status_code=400, 
            content={"status": "error", "detail": f"Erro na execução SQL: {str(e)}"}
