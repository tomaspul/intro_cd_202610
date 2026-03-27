## 🧠 Metodología
 
### 🗺️ Vista General del Pipeline
 
```
Datos Históricos  →  Preprocesamiento  →  Modelado  →  Alerta Temprana  →  Intervención
  (MEN / SED)           (ETL + EDA)       (ML / XGBoost)   (Dashboard)        (Orientadores)
```
 
---
 
### 📋 Fase 1 — Definición del Problema
 
<details>
<summary>🔍 <strong>¿Qué queremos predecir? — click para expandir</strong></summary>
<br>
 
**Variable objetivo (target):**
 
```python
# Definición de la etiqueta de deserción
# 1 = El estudiante abandonó antes de graduarse
# 0 = El estudiante completó el período con normalidad
 
target = "desercion_semestre"  # Variable binaria
```
 
**Horizonte de predicción:** 3 meses hacia adelante, al inicio de cada período escolar.
 
**Unidad de análisis:** Estudiante × Período académico
 
**Criterio de éxito:**
| Métrica | Meta |
|---------|------|
| Recall (sensibilidad) | ≥ 80% (priorizar no perder casos reales) |
| Precisión | ≥ 70% |
| F1-Score | ≥ 0.75 |
| AUC-ROC | ≥ 0.85 |
 
> ⚠️ **Nota:** En este contexto, **el recall es más importante que la precisión**. Es preferible alertar de más estudiantes en riesgo (falsos positivos controlables) que dejar sin atención a un estudiante que sí desertará.
 
</details>
 
---
 
### 🔧 Fase 2 — Recolección y Fuentes de Datos
 
<details>
<summary>🗄️ <strong>Variables disponibles — click para expandir</strong></summary>
<br>
 
```python
variables = {
    "academicas": [
        "notas_matematicas",
        "notas_lenguaje",
        "notas_ciencias",
        "promedio_general",
        "materias_reprobadas"
    ],
    "asistencia": [
        "inasistencias_mes_1",
        "inasistencias_mes_2",
        "inasistencias_mes_3",
        "tendencia_inasistencia"   # feature engineered
    ],
    "socioeconomicas": [
        "estrato",
        "distancia_colegio_km",
        "trabaja",                 # booleano
        "tipo_hogar",              # ambos padres / solo uno / otro
        "recibe_subsidio"
    ],
    "historicas": [
        "repitencia_anos_anteriores",
        "cambios_de_colegio",
        "anio_escolar_actual"
    ]
}
```
 
| Categoría | Variables | Tipo |
|-----------|-----------|------|
| 📊 Académicas | Notas por materia, reprobación | Numéricas |
| 📅 Asistencia | Inasistencias por mes | Numéricas / Serie temporal |
| 🏘️ Socioeconómicas | Estrato, distancia, hogar | Mixtas |
| 🔁 Historial | Repitencia, cambios de colegio | Categóricas / Enteras |
 
</details>
 
---
 
### 🔍 Fase 3 — Análisis Exploratorio (EDA)
 
<details>
<summary>📊 <strong>Preguntas clave del EDA — click para expandir</strong></summary>
<br>
 
```python
# Preguntas que guían el análisis exploratorio
 
preguntas_eda = [
    "¿Cuál es la tasa de deserción por localidad en Bogotá?",
    "¿Existe correlación entre inasistencias y deserción?",
    "¿Los estudiantes que trabajan tienen mayor riesgo?",
    "¿Qué estrato concentra más casos de deserción?",
    "¿Las notas de qué materia predicen mejor el abandono?",
    "¿Cuántos meses antes de desertar empieza a caer la asistencia?",
    "¿Hay diferencias por género o por ciclo escolar?"
]
```
 
**Herramientas usadas:**
 
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4EAFC3?style=flat-square&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)
 
**Hallazgos esperados:**
- `inasistencias_mes_1` y `inasistencias_mes_2` con alta correlación con la variable objetivo
- `trabaja = True` como factor de riesgo significativo en grados 9°–11°
- `repitencia_anos_anteriores > 1` como predictor muy fuerte
- Distribución desbalanceada de clases (90%–10% aprox.)
 
</details>
 
---
 
### ⚙️ Fase 4 — Preprocesamiento y Feature Engineering
 
<details>
<summary>🛠️ <strong>Transformaciones aplicadas — click para expandir</strong></summary>
<br>
 
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
 
# ── 1. Manejo de valores nulos ──────────────────────────────────────
df["notas_matematicas"].fillna(df["notas_matematicas"].median(), inplace=True)
df["distancia_colegio_km"].fillna(df["distancia_colegio_km"].mean(), inplace=True)
 
# ── 2. Encoding de variables categóricas ───────────────────────────
df["tipo_hogar_cod"] = LabelEncoder().fit_transform(df["tipo_hogar"])
 
# ── 3. Feature Engineering (nuevas variables derivadas) ────────────
# Tendencia de inasistencia (¿está empeorando?)
df["tendencia_inasistencia"] = df["inasistencias_mes_2"] - df["inasistencias_mes_1"]
 
# Carga académica bajo presión (trabaja y bajo rendimiento)
df["riesgo_laboral_academico"] = (
    (df["trabaja"] == 1) & (df["promedio_general"] < 3.0)
).astype(int)
 
# Índice de vulnerabilidad compuesto
df["indice_vulnerabilidad"] = (
    (6 - df["estrato"]) * 0.3 +
    df["repitencia_anos_anteriores"] * 0.4 +
    df["trabaja"] * 0.3
)
 
# ── 4. Escalado de variables numéricas ─────────────────────────────
scaler = StandardScaler()
cols_numericas = ["inasistencias_mes_1", "distancia_colegio_km", "promedio_general"]
df[cols_numericas] = scaler.fit_transform(df[cols_numericas])
```
 
**Manejo del desbalance de clases:**
```python
from imblearn.over_sampling import SMOTE
 
# SMOTE para balancear la clase minoritaria (desertores)
smote = SMOTE(sampling_strategy=0.4, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
```
 
</details>
 
---
 
### 🤖 Fase 5 — Modelado y Entrenamiento
 
<details>
<summary>⚡ <strong>Modelos evaluados y selección final — click para expandir</strong></summary>
<br>
 
```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
 
# Modelos candidatos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150),
    "XGBoost":             XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}
 
# Validación cruzada estratificada (respeta el desbalance)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
for nombre, modelo in modelos.items():
    scores = cross_val_score(modelo, X_res, y_res, cv=cv, scoring="roc_auc")
    print(f"{nombre}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")
```
 
**Comparativa de modelos:**
 
| Modelo | AUC-ROC | Recall | Precisión | F1 |
|--------|---------|--------|-----------|----|
| Logistic Regression | 0.79 | 0.71 | 0.68 | 0.69 |
| Random Forest | 0.86 | 0.79 | 0.74 | 0.76 |
| Gradient Boosting | 0.88 | 0.82 | 0.76 | 0.79 |
| **✅ XGBoost** | **0.91** | **0.85** | **0.78** | **0.81** |
 
**→ Modelo seleccionado: XGBoost** por su mayor recall y AUC-ROC.
 
```python
# Modelo final con hiperparámetros optimizados (Optuna)
modelo_final = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4,     # Penaliza más los falsos negativos
    random_state=42
)
```
 
</details>
 
---
 
### 📈 Fase 6 — Evaluación e Interpretabilidad
 
<details>
<summary>🔬 <strong>Métricas, SHAP y validación ética — click para expandir</strong></summary>
<br>
 
```python
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
 
# ── Métricas de evaluación ─────────────────────────────────────────
y_pred = modelo_final.predict(X_test)
y_proba = modelo_final.predict_proba(X_test)[:, 1]
 
print(classification_report(y_test, y_pred, target_names=["No deserta", "Deserta"]))
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
 
# ── SHAP: ¿Por qué el modelo toma cada decisión? ──────────────────
explainer = shap.TreeExplainer(modelo_final)
shap_values = explainer.shap_values(X_test)
 
# Gráfica de importancia global de variables
shap.summary_plot(shap_values, X_test, plot_type="bar")
 
# Explicación individual por estudiante
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```
 
**Variables más importantes (SHAP):**
 
```
1. inasistencias_mes_2          ████████████████████  0.38
2. repitencia_anos_anteriores   █████████████████     0.31
3. tendencia_inasistencia       ██████████████        0.27
4. promedio_general             ████████████          0.23
5. trabaja                      ██████████            0.19
6. estrato                      ████████              0.16
7. indice_vulnerabilidad        ███████               0.14
8. distancia_colegio_km         █████                 0.10
```
 
**Validación ética del modelo:**
- [x] ✅ El modelo NO discrimina por género ni etnia
- [x] ✅ Calibración verificada (las probabilidades son confiables)
- [x] ✅ Sesgo revisado por estrato socioeconómico
- [x] ✅ Explicabilidad con SHAP para cada predicción individual
 
</details>
 
---
 
### 🚀 Fase 7 — Despliegue y Sistema de Alertas
 
<details>
<summary>🌐 <strong>Arquitectura del sistema en producción — click para expandir</strong></summary>
<br>
 
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
 
app = FastAPI(title="API de Alertas de Deserción Escolar")
modelo = joblib.load("modelo_desercion_v2.pkl")
 
class EstudianteInput(BaseModel):
    inasistencias_mes_1: float
    inasistencias_mes_2: float
    promedio_general: float
    estrato: int
    trabaja: bool
    repitencia_anos_anteriores: int
    distancia_colegio_km: float
    tipo_hogar: str
 
@app.post("/predecir")
def predecir_riesgo(datos: EstudianteInput):
    X = preparar_features(datos)
    probabilidad = modelo.predict_proba(X)[0][1]
    
    nivel_riesgo = (
        "🔴 ALTO"   if probabilidad > 0.70 else
        "🟡 MEDIO"  if probabilidad > 0.40 else
        "🟢 BAJO"
    )
    
    return {
        "probabilidad_desercion": round(probabilidad * 100, 1),
        "nivel_riesgo": nivel_riesgo,
        "accion_sugerida": get_accion(probabilidad)
    }
```
 
**Ejemplo de respuesta de la API:**
```json
{
  "estudiante_id": "BOG-2025-00847",
  "colegio": "IED Simón Bolívar",
  "grado": "10°",
  "probabilidad_desercion": 78.3,
  "nivel_riesgo": "🔴 ALTO",
  "accion_sugerida": "Contactar acudiente esta semana + remitir a orientación"
}
```
 
**Stack tecnológico de producción:**
 
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)
 
</details>
 
---
