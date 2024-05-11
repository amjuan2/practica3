import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Descargar el conjunto de datos
url = "https://www.datos.gov.co/resource/usdv-hgxz.json"
response = requests.get(url)
data = response.json()

# Convertir los datos a un DataFrame de Pandas
df = pd.DataFrame(data)

# Preprocesar los datos
# Convertir la columna '_1_edad' a tipo numérico
df['_1_edad'] = pd.to_numeric(df['_1_edad'], errors='coerce')
# Eliminar filas con valores de edad faltantes o negativos
df = df.dropna(subset=['_1_edad'])
df = df[df['_1_edad'] >= 0]
# Convertir el género a variables dummy
df['_2_g_nero'] = df['_2_g_nero'].map({'M': 0, 'F': 1})
# Convertir el conocimiento de anticonceptivos a tipo numérico
df['_14_qu_m_todos_anticonceptivos'] = pd.to_numeric(df['_14_qu_m_todos_anticonceptivos'], errors='coerce')
# Eliminar filas con valores de conocimiento de anticonceptivos faltantes
df = df.dropna(subset=['_14_qu_m_todos_anticonceptivos'])

# Dividir los datos en características (X) y etiquetas (y)
X = df[['_1_edad', '_2_g_nero']].values
y = df['_14_qu_m_todos_anticonceptivos'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['_2_g_nero'])

# Verificar el tamaño del conjunto de entrenamiento
print("Tamaño del conjunto de entrenamiento:", len(X_train))

# Crear el modelo de redes neuronales con Keras
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))  # Capa oculta con 64 neuronas y función de activación ReLU
model.add(Dense(1, activation='linear'))  # Capa de salida con una neurona y función de activación lineal

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.2)

# Evaluar el modelo
loss, mse = model.evaluate(X_test, y_test, verbose=0)
print("Error cuadrático medio en el conjunto de prueba:", mse)

# Graficar la pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Pérdida durante el entrenamiento')
plt.show()
