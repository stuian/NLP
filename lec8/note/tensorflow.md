## Tensorflow

### 一、资源链接 ###

- github链接：[https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

- tensorflow教程：[http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html](http://www.tensorfly.cn/tfdoc/get_started/basic_usage.html)

### 二、基础安装 ###

安装环境：windows7 64位

1、CPU版：直接pip install tensorflow

> (如果电脑有python2和python3——改成py -2/-3 -m pip install tensorflow)

2、参考链接（GPU版本）：[http://blog.csdn.net/u010099080/article/details/53418159](http://blog.csdn.net/u010099080/article/details/53418159)

> 1、安装CPU版本需要 CUDA 和 cuDNN 的支持，CPU版本的不需要

> 2、去下面这个网站检验你的GPU显卡是否支持CUDA：[https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)


**Question**（目前安装的是CPU版本）：

1、

    ModuleNotFoundError: No module named 'tensorflow.python'

> 解决方法：重装


###三、学习进度 ###

- 莫烦教程tensorflow 12
- Tensorflow中文社区完整教程Mnist进阶

> 因为python语法我很早就熟悉了，学起来没那么费劲，主要是numpy和tensorflow的语法学习。

#### 3.1 笔记总结 ####

![](https://github.com/stuian/NLP/blob/master/lec8/note/IMG20171016064841.jpg?raw=true)

![](https://github.com/stuian/NLP/blob/master/lec8/note/IMG20171016065019.jpg?raw=true)

![](https://github.com/stuian/NLP/blob/master/lec8/note/IMG20171016065102.jpg?raw=true)

#### 3.2 numpy ####

np.random.randn()

np.random.rand()

np.random.normal()

np.linspace

3.3 tensorflow

**1、varible**

example code:

    import tensorflow as tf
    
    state = tf.Variable(0,name='counter')
    print(state.name)
    
    one = tf.constant(1)
    new_value = tf.add(state,one)
    update = tf.assign(state,new_value)
    
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
    sess.run(update)
    print(sess.run(state))

**2、session**

example code:

    import tensorflow as tf
    
    matrix1 = tf.constant([[3,3]])
    matrix2 = tf.constant([[2],
      [2]])
    
    product = tf.matmul(matrix1,matrix2)
    
    #method1
    # sess = tf.Session()
    # result = sess.run(product)
    # print(result)
    # sess.close()
    
    #method2
    with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    
**3、placeholder**

example code:

    import tensorflow as tf
    
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    result = tf.multiply(x,y)
    
    with tf.Session() as sess:
    print(sess.run(result,feed_dict={x:7.0,y:2.0}))

**4、一层神经网络（拟合直线y=0.1x+0.3）**

example code:

    import tensorflow as tf
    import numpy as np
    
    #create data
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*0.1+0.3
    
    #create tensorflow structure start
    Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
    biases = tf.Variable(tf.zeros([1]))
    
    y = Weights*x_data+biases
    
    loss = tf.reduce_mean(tf.square(y-y_data))
    optimizer= tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    
    init = tf.initialize_all_variables()
    #create tensorflow structure start
    
    sess = tf.Session()
    sess.run(init)
    
    for step in range(201):
    sess.run(train)
    if step % 20 == 0:
    print(step,sess.run(Weights),sess.run(biases))
    
**5、添加一层神经层 **

example code:

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
    
    

**6、三层神经网络（拟合直线y=x^2-0.5）**

example code:

    import tensorflow as tf
    import numpy as np
    
    def add_layer(inputs,in_size,out_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1) #?0.1
    wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
    outputs = wx_plus_b
    else:
    outputs = activation_function(wx_plus_b)
    return outputs
    
    #所有的数据
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)
    y_data = np.square(x_data)-0.5+noise
    
    #传入tensorflow的数据
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])
    
    l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
    prediction = add_layer(l1,10,1,activation_function=None)
    
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    
    for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
    print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

