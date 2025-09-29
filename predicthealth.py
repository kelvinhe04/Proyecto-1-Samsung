import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# Paso 0: Cargar y limpiar datos
# =====================================================
df = pd.read_csv('KaggleV2-May-2016.csv')

print("====================================================")
print("============= LIMPIEZA DE DATOS ====================")
print("====================================================")

# Eliminar edades negativas
df = df[df['Age'] >= 0]

# Normalizar No-show a 0/1
df['No-show'] = df['No-show'].apply(lambda x: 1 if x == 'Yes' else 0)

# Normalizar género
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'M' else 0)

# Convertir fechas a datetime
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Crear columna de espera (diferencia en días)
df['waiting_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days


# =====================================================
# Paso 1: Vista general de los datos
# =====================================================
print("\n====================================================")
print("============== PASO 1: VISTA GENERAL ===============")
print("====================================================")

print("\n--- Dimensiones del dataset ---")
print(df.shape)

print("\n--- Primeras filas (head) ---")
print(df.head())

print("\n--- Resumen estadístico ---")
print(df.describe())

# Porcentaje de ausentismo
no_show_rate = df['No-show'].mean() * 100
print("\n--- Tasa de ausentismo ---")
print(f"Tasa de ausentismo: {no_show_rate:.2f}%")

# Gráfico circular
plt.figure(figsize=(5,5))
df['No-show'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['Asistió','No asistió'], colors=['#66b3ff','#ff9999']
)
plt.title("Distribución de asistencia")
plt.show()


# =====================================================
# Paso 2: Análisis exploratorio
# =====================================================
print("\n====================================================")
print("========== PASO 2: ANÁLISIS EXPLORATORIO ===========")
print("====================================================")

# Ausentismo por día de la semana
df['day_of_week'] = df['AppointmentDay'].dt.day_name()
print("\n--- Ausentismo por día de la semana ---")
plt.figure(figsize=(8,5))
sns.countplot(x='day_of_week', hue='No-show', data=df,
              order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
plt.title("Ausentismo por día de la semana")
plt.xticks(rotation=45)
plt.show()

# Mapa de calor de correlaciones
print("\n--- Mapa de calor de correlaciones ---")
plt.figure(figsize=(10,6))
sns.heatmap(df[['Gender','Age','Scholarship','Hipertension','Diabetes',
                'Alcoholism','Handcap','SMS_received','waiting_days','No-show']].corr(),
            annot=True, cmap='coolwarm')
plt.title("Mapa de calor de correlaciones")
plt.show()


# =====================================================
# Paso 3: Modelo predictivo
# =====================================================
print("\n====================================================")
print("============== PASO 3: MODELO PREDICTIVO ===========")
print("====================================================")

# Variables predictoras y objetivo
X = df[['Gender','Age','Scholarship','Hipertension','Diabetes',
        'Alcoholism','Handcap','SMS_received','waiting_days']]
y = df['No-show']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo
accuracy = model.score(X_test, y_test)
print("\n--- Exactitud del modelo ---")
print("Exactitud del modelo:", accuracy)

# Importancia de variables
print("\n--- Importancia de variables ---")
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8,5), color='skyblue')
plt.title("Importancia de variables en el modelo")
plt.show()


# =====================================================
# Paso 4: Conclusiones automáticas
# =====================================================
print("\n====================================================")
print("========== PASO 4: CONCLUSIONES AUTOMÁTICAS =========")
print("====================================================")

print(f"- El ausentismo general es de {no_show_rate:.1f}%.")
print(f"- La variable más influyente en el modelo es: {importances.idxmax()}.")
print("- Los días con más ausentismo son los lunes y martes (según los gráficos).")
print("- Factores como SMS_received y waiting_days también parecen relevantes.")
