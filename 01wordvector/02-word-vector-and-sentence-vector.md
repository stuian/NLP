## word-vector and sentence-vector ##

### 1 one-hot编码

没什么好说的。

### 2 基于svd的方法

遍历所有的文本数据集，然后统计词出现的次数，接着用一个矩阵X来表示所有的次数情况，紧接着对X进行奇异值分解得到一个USVT的分解。然后用U的行（rows）作为所有词表中词的词向量。对于矩阵X，我们有几种选择，咱们一起来比较一下。

### 2.1词文档矩阵

最终所得矩阵维度为：V（词汇数量） x M(文档个数)。维度超大，必须要优化。这里的文档是一篇文章为单位还是一个句子为单位取决于数据量大小。

### 2.2基于窗口的词词共现

与2.1不同，换一种计数方法，我们先规定一个固定大小的窗口，然后统计每个词出现在窗口中次数，最终得到一个相关性矩阵。

![](https://github.com/stuian/NLP-CS224d/blob/master/01wordvector/pictures/words-matrix.jpg?raw=true)


由于这个矩阵一般很大，我们对X做奇异值分解，观察观察奇异值（矩阵的对角元素），并根据我们期待保留的百分比来进行阶段（只保留前k个维度）：

（或者说其中s是对矩阵x的奇异值分解。s除了对角元素不为0，其他元素都为0，并且对角元素从大到小排列。s中有n个奇异值，一般排在后面的比较接近0，所以仅保留比较大的r个奇异值。）

![](https://github.com/stuian/NLP-CS224d/blob/master/01wordvector/pictures/svd.jpg?raw=true)

> 关于[svd](https://blog.csdn.net/YE1215172385/article/details/79414702)
> [用svd处理推荐系统矩阵](https://blog.csdn.net/qq_36523839/article/details/82347332)

![](https://img-blog.csdn.net/20180301170056165?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvWUUxMjE1MTcyMzg1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 3 基于迭代-word2vector

### 3.1 基础-nnlm（神经网络语言模型）

nnlm以n-grams为基础：



![](https://github.com/stuian/NLP-CS224d/blob/master/01wordvector/pictures/nnlm.jpg?raw=true)





