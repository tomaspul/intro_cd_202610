<div align="center">
 
---
 
## 🔬 Metodología
 
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Comfortaa&size=22&pause=1200&color=0d4a2a&center=true&vCenter=true&width=1000&height=60&lines=¿Cómo+le+decimos+a+los+datos+que+nos+cuenten+una+historia%3F;7+pasos+para+anticipar+la+deserción+escolar+📚)](https://git.io/typing-svg)
 
> *Cada fase es un escalón. Juntas, convierten filas en una base de datos en alertas que salvan trayectorias de vida.*
 
---
 
### Fase 1 · Entendimiento del Problema 🧭
 
<img src="https://img.shields.io/badge/Etapa-Estratégica-0d4a2a?style=for-the-badge&logoColor=white"/>
 
Antes de tocar un solo dato, nos preguntamos: **¿qué queremos resolver realmente?**
 
El Ministerio de Educación necesita anticiparse — no reaccionar. Aquí definimos la pregunta central del proyecto, el tipo de solución (un modelo de clasificación), quiénes van a usar los resultados (orientadores escolares), y qué significaría que el modelo funcione bien. También establecemos el horizonte de predicción: **3 meses antes** de que ocurra la deserción.
 
```
🎯 Pregunta clave:
¿Puede un modelo de datos identificar, al inicio del período,
qué estudiante tiene riesgo de no terminar el año?
```
 
---
 
### Fase 2 · Recolección de Datos 🗄️
 
<img src="https://img.shields.io/badge/Etapa-Fundacional-1a6b3c?style=for-the-badge&logoColor=white"/>
 
Los datos vienen del **Sistema Educativo Distrital (SED)** y el **Ministerio de Educación Nacional (MEN)**. Aquí identificamos qué registros históricos tenemos disponibles y cuáles variables son relevantes para el problema.
 
Trabajamos con información de **42.500 estudiantes** de bachillerato en **214 colegios públicos** de Bogotá entre 2019 y 2023. Cada registro es una historia: cuánto faltó, cómo le fue, dónde vive, con quién vive.
 
```
📦 Fuentes principales:
   ├── SIMAT (Sistema Integrado de Matrícula)
   ├── Registros de calificaciones por período
   ├── Encuestas socioeconómicas estudiantiles
   └── Datos de geolocalización por localidad
```
 
---
 
### Fase 3 · Exploración de los Datos (EDA) 🔍
 
<img src="https://img.shields.io/badge/Etapa-Descubrimiento-20b267?style=for-the-badge&logoColor=white"/>
 
Antes de construir cualquier modelo, **escuchamos lo que los datos tienen para decir**. En esta fase hacemos preguntas, graficamos distribuciones, buscamos patrones y encontramos los primeros indicios de qué variables importan más.
 
Algunos hallazgos que esperamos descubrir:
 
| 🔎 Pregunta | 📊 Lo que buscamos |
|---|---|
| ¿Cuándo empieza a caer la asistencia? | Meses previos a la deserción |
| ¿Qué localidades concentran más casos? | Mapas de riesgo por zona |
| ¿Trabajar afecta el rendimiento? | Relación trabajo–notas–ausencias |
| ¿Qué tan desbalanceados están los datos? | % de desertores vs. no desertores |
 
> 💡 *El EDA no es un trámite. Es donde el analista aprende a ver lo que los números esconden.*
 
---
 
### Fase 4 · Preparación de los Datos ⚙️
 
<img src="https://img.shields.io/badge/Etapa-Transformación-0d4a2a?style=for-the-badge&logoColor=white"/>
 
Los datos rara vez llegan limpios. En esta fase los transformamos para que el modelo pueda entenderlos. Tratamos valores faltantes, convertimos categorías en números, normalizamos escalas y creamos **nuevas variables** que capturen señales que las originales no muestran solas.
 
Por ejemplo: un estudiante con inasistencias crecientes mes a mes es más riesgoso que uno con el mismo total distribuido uniformemente. Esa *tendencia* es una variable nueva que construimos aquí.
 
```
🛠️ Lo que hacemos en esta fase:
   ├── Limpieza de valores nulos o inconsistentes
   ├── Codificación de variables categóricas
   ├── Creación de variables derivadas (Feature Engineering)
   ├── Normalización de escalas numéricas
   └── Balanceo de clases (los desertores son minoría en los datos)
```
 
---
 
### Fase 5 · Modelado 🤖
 
<img src="https://img.shields.io/badge/Etapa-Núcleo_del_Proyecto-1a6b3c?style=for-the-badge&logoColor=white"/>
 
Aquí entrenamos el modelo que aprenderá a identificar patrones de riesgo a partir de los datos históricos. Probamos varios algoritmos de clasificación y comparamos cuál se desempeña mejor para nuestro objetivo.
 
Dado que nos importa **no perder casos reales de deserción**, priorizamos modelos con alta sensibilidad — preferiríamos alertar de más que dejar sin atención a un estudiante en riesgo.
 
| 🧠 Modelo | ¿Por qué lo probamos? |
|---|---|
| Regresión Logística | Línea base sencilla e interpretable |
| Random Forest | Robusto y maneja variables mixtas |
| Gradient Boosting | Excelente para datos desbalanceados |
| XGBoost ✅ | Mayor precisión y control del sesgo |
 
> *Elegimos el modelo como elegimos un médico: no solo el más sofisticado, sino el más confiable para este caso específico.*
 
---
 
### Fase 6 · Evaluación e Interpretabilidad 📊
 
<img src="https://img.shields.io/badge/Etapa-Validación-20b267?style=for-the-badge&logoColor=white"/>
 
Un modelo que funciona pero nadie entiende no sirve en la práctica. En esta fase medimos qué tan bien predice el modelo y, sobre todo, **explicamos por qué toma cada decisión**.
 
Usamos **SHAP** (SHapley Additive exPlanations), una técnica que permite ver cuánto aportó cada variable a la predicción de un estudiante específico. Así un orientador puede saber: *"este alumno está en riesgo principalmente porque faltó 14 días en el último mes y trabaja"*.
 
```
📏 Métricas que evaluamos:
   ├── AUC-ROC     → ¿Qué tan bien distingue entre casos?
   ├── Recall      → ¿Cuántos desertores reales detectamos?
   ├── Precisión   → ¿Cuántas alertas son correctas?
   ├── F1-Score    → Balance entre las dos anteriores
   └── Sesgo       → ¿El modelo es justo entre estratos y géneros?
```
 
---
 
### Fase 7 · Despliegue y Sistema de Alertas 🚀
 
<img src="https://img.shields.io/badge/Etapa-Impacto_Real-0d4a2a?style=for-the-badge&logoColor=white"/>
 
El modelo no vive en un notebook. Vive en manos de quienes pueden actuar. En esta última fase construimos el sistema que pone las predicciones al servicio de los orientadores escolares, de forma simple y accionable.
 
Al inicio de cada período, el sistema genera una lista priorizada por colegio con el nivel de riesgo de cada estudiante y una sugerencia de intervención: llamar al acudiente, gestionar subsidio de transporte, remitir a apoyo psicosocial.
 
| 🟢 Riesgo Bajo | 🟡 Riesgo Medio | 🔴 Riesgo Alto |
|---|---|---|
| Monitoreo mensual | Contacto con familia | Intervención inmediata |
| Seguimiento de asistencia | Apoyo académico dirigido | Orientación + subsidios |
 
```
🌐 Herramientas del sistema:
   ├── API de predicción en tiempo real
   ├── Dashboard interactivo para orientadores
   ├── Reportes automáticos por colegio
   └── Alertas semanales por correo / SMS
```
 
> *La ciencia de datos no termina cuando el modelo aprende. Termina cuando un estudiante no abandona.*
 
---
 
<br/>
 
[![Typing SVG](https://readme-typing-svg.demolab.com?font=Comfortaa&size=18&pause=1500&color=20b267&center=true&vCenter=true&width=1000&height=50&lines=Fase+1+🧭+→+Fase+2+🗄️+→+Fase+3+🔍+→+Fase+4+⚙️+→+Fase+5+🤖+→+Fase+6+📊+→+Fase+7+🚀)](https://git.io/typing-svg)
 
</div>
