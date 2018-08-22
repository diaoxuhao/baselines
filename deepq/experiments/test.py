import tensorflow as tf


a = tf.linspace(0.,1.,10)
a = tf.expand_dims(a,0)

b = tf.transpose(a)

print(tf.matmul(a,b))
