import numpy as np
import ejercicioLibro as ejer

def vectorize_sequences(sequences, dimension=10000):
     results = np.zeros((len(sequences), dimension))
     for i, sequence in enumerate(sequences):
         for j in sequence:
            results[i, j] = 1.
     return results
x_train = vectorize_sequences(ejer.train_data)
x_test = vectorize_sequences(ejer.test_data)

y_train = np.asarray(ejer.train_labels).astype("float32")
y_test = np.asarray(ejer.test_labels).astype("float32")
print(x_train[0])