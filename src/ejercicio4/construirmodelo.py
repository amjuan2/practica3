import preparardata as pd
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
 layers.Dense(16, activation="relu"),
 layers.Dense(16, activation="relu"),
 layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop",
 loss="binary_crossentropy",
 metrics=["accuracy"])

x_val = pd.x_train[:10000]
partial_x_train = pd.x_train[10000:]
y_val = pd.y_train[:10000]
partial_y_train = pd.y_train[10000:]

history = model.fit(partial_x_train,
 partial_y_train,
 epochs=20,
 batch_size=512,
 validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())
