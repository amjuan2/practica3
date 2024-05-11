import construirmodelo as cm
import preparardata as pd
import keras
from tensorflow.keras import layers


cm.model = keras.Sequential([
 layers.Dense(16, activation="relu"),
 layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

cm.model.compile(optimizer="rmsprop",
 loss="binary_crossentropy",
 metrics=["accuracy"])
cm.model.fit(pd.x_train, pd.y_train, epochs=4, batch_size=512)
results = cm.model.evaluate(pd.x_test, pd.y_test)

print(results)

print(cm.model.predict(pd.x_test))