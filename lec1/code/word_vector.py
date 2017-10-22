import numpy as np
import tensorflow as tf
corpus_raw = 'He is the king.The king is royal.She is the royal queen'

#convert to lower case
corpus_raw = corpus_raw.lower()

#create a list to determine the index of every word
words = []
for word in corpus_raw.split():
    if word != '.':
        words.append(word)
words = set(words) #remove the duplicate words
word2int = {}
int2word = {}
vocab_size = len(words)
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word
# create dicts of index and words

#convert sentence to words
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    sentences.append(sentence.split())
#every element of sentences is the words of a sentence

#create traing data
data = []

WINDOW_SIZE = 2

for sentence in sentences:
    for word_index,word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE,0) : min(word_index + WINDOW_SIZE,len(sentence)) + 1]:
            #the reason of adding 1 is that ending isn't being taken into consideration
            if nb_word != word:
                data.append([word,nb_word])
                # the couple of input and output

#convert couple of word to numbers ,then continue to turn from numbers to one-hot
def to_one_hot(data_point_index,vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp

x_train = []
y_train = []
print(x_train)
for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]],vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]],vocab_size))

#convert x and y to nunmpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

#make tensorflow mode
x = tf.placehodler(tf.float32,shape=(None,vocab_size))
y_label = tf.placeholder(tf.float32,shape=(None,vocab_size))

EMBEDDING_DIM = 5
w1 = tf.Variable(tf.random_normal([vocab_size,EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))

hidden_representation = tf.add(tf.matmul(x,w1),b1)

#deal with hidden layer
#predice the surrounding words
w2 = tf.Variable(tf.random_normal([EMBEDDING_DIM,vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation,w2),b2))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000

for _ in range(n_iters):
    sess.run(train_step,feed_dict={x:x_train,y_label:y_train})
    print('loss is :',sess.run(cross_entropy_loss,feed_dict={x:x_train,y_label:y_train}))
