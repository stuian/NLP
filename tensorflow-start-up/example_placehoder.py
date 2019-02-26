import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
result = tf.multiply(x,y)

with tf.Session() as sess:
    print(sess.run(result,feed_dict={x:7.0,y:2.0}))