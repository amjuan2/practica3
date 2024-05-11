import numpy as np
import tensorflow as tf

#tensor all ones
x = np.ones(shape=(2, 2))
# 3.3 numpy array es asignable
x[0, 0] = 0.

# 3.4 asignacion falla porque tensor no es asignable
x = tf.ones(shape=(2, 2))
x[0, 0] = 0.

