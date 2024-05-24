# Permite hacer solicitudes HTTP
import requests
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Cargar el modelo IMDB preentrenado
(train_data, _), (_, _) = imdb.load_data(num_words=100000)
word_index = imdb.get_word_index()

# Define un modelo
model = keras.Sequential([
 layers.Dense(32, activation="relu"),
 layers.Dense(16, activation="relu"),
 layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
 loss="binary_crossentropy",
 metrics=["accuracy"])

# Carga los datos desde la URL
url = "https://www.datos.gov.co/resource/r5hj-eu84.json"
# Realiza una solicitud HTTP GET
response = requests.get(url)
data = response.json()

# Preprocesar el conjunto de datos
def prep_data(data, max_len=100000):
    processed_comments = []
    for item in data:
        comentario = item["comentario"]
        # Tokeniza el comentario dividiéndolo en palabras y asignando a
        # cada palabra su índice correspondiente del diccionario word_index.
        sequence = [word_index.get(word.lower(), 2) for word in comentario.split()]
        # Reemplaza los índices mayores a 9999 con 9999
        # para asegurarse de que todos los índices estén dentro del rango permitido.
        sequence = [x if x < 100000 else 9999 for x in sequence]
        # Agregar valores adicionales a una secuencia para que tenga una longitud específica.
        pad_sequence = pad_sequences([sequence], maxlen=max_len)
        # Agrega la secuencia a la lista de comentarios procesados.
        processed_comments.append(pad_sequence)
    #Convierte la lista de comentarios procesados en una matriz y la devuelve.
    return np.vstack(processed_comments)

# Transformar comentarios en vectores
def vectorize_sequences(sequences, dimension=100000):
    # Crea una matriz de ceros
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
    # Establece el valor correspondiente en la matriz
        results[i, sequence] = 1.
    return results

# Preprocesar los datos
x_data = prep_data(data)
x_data = vectorize_sequences(x_data)

# Realizar predicciones
predictions = model.predict(x_data)

# Mostrar resultados
for i, item in enumerate(data):
    comentario = item["comentario"]
    # Convertir a porcentaje
    positividad = predictions[i][0] * 100
    print(f"Comentario: {comentario}")
    print("Sentimiento:", "Positivo" if predictions[i] > 0.5 else "Negativo")
    '''if predictions[i] < 0.45:
        print("Sentimiento:", "Negativo")
    elif 0.45 <= predictions[i] <= 0.55:
        print("Sentimiento:", "Neutral")
    else:
        print("Sentimiento:", "Positivo")'''
    print("Porcentaje de Positividad:", f"{positividad:.2f}%")
    print()