import tensorflow as tf

#两个张量
X = tf.placeholder("float",[None,784])
y_ = tf.placeholder("float",[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(X,W)+b)

#所有图片的交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#学习率为0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_tables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    #该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict = {x:batch_xs,y:batch_ys})

#为什么是1，预测的概率不一定最大是1，所有概率加起来是1吧
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.case(correct_prediction,"float"))

print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnnist.test.labels})