# Proyecto de Predicción de Ausentismo en Citas Médicas

## Introducción
Este proyecto utiliza un modelo de aprendizaje automático (Random Forest) para predecir si un paciente asistirá o no a su cita médica, basado en características como edad, género, condiciones médicas, días de espera, entre otros.

## Requisitos
Antes de comenzar, asegúrate de tener instalados los siguientes programas y bibliotecas:

### Software necesario:
- Python 3.8 o superior
- Un entorno virtual (opcional pero recomendado)

### Bibliotecas de Python:
Instala las siguientes bibliotecas ejecutando:
```bash
pip install -r requirements.txt
```
El archivo `requirements.txt` debe incluir:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Pasos para comenzar

### 1. Clonar el repositorio
Si estás trabajando con un repositorio, clónalo usando:
```bash
git clone <URL-del-repositorio>
```

### 2. Crear un entorno virtual
Es recomendable usar un entorno virtual para evitar conflictos entre dependencias:
```bash
python -m venv venv
source venv/Scripts/activate  # En Windows
source venv/bin/activate      # En Linux/Mac
```

### 3. Instalar dependencias
Ejecuta el siguiente comando para instalar las bibliotecas necesarias:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el script
Corre el archivo principal `predicthealth.py` para entrenar el modelo y generar los análisis:
```bash
python predicthealth.py
```

## Archivos principales
- `predicthealth.py`: Contiene el código principal para el análisis y entrenamiento del modelo.
- `KaggleV2-May-2016.csv`: Dataset utilizado para el análisis.
- `README.md`: Este archivo con instrucciones.

## Resultados esperados
1. Gráficos exploratorios:
   - Distribución de asistencia y ausentismo.
   - Ausentismo por día de la semana.
   - Mapa de calor de correlaciones.
2. Entrenamiento del modelo Random Forest.
3. Evaluación del modelo con una métrica de exactitud.

## Notas adicionales
- Asegúrate de que el archivo `KaggleV2-May-2016.csv` esté en la misma carpeta que el script principal.
- Si deseas mejorar el modelo, puedes ajustar los hiperparámetros del Random Forest o probar otros algoritmos.

## Contacto
Si tienes preguntas o sugerencias, no dudes en contactarme.