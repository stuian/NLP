import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #?0.1
    wx_plus_b = tf.matmul(input_value,Weights) + biases
    if activation_function is None:
        output_value = wx_plus_b
    else:
        output_value = activation_function(wx_plus_b)
    return outputs

