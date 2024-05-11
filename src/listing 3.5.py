import numpy as np
import tensorflow as tf

v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
print(v)

#3.6 assign a value to a tf vsriable
v.assign(tf.ones((3, 1)))
print(v)

# 3.7 asignando a subvalor de un tf variable
v[0, 0].assign(3.)
print(v)