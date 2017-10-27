## lec9 用Recursive NN学习语法剖析树和句子向量表示

- sentence进行vector表示
- parsingg（语法分析）
- 构建object function的方法max-margin
- BPTS
- recursive NN

对于词级别的向量表示，常用的有分布式表示，布朗聚类，可以作为输入特征用到模型中去，但是难以表示更长的短语、句子。对于文档级别的向量表示，常用的有词袋模型，PCA，在信息检索上很有用，但是略去了众多细节和词语顺序。

### 语言的语法规则具有很强的递归性。 ###

之前已经通过训练一元或n元语言模型得到的词向量。事实上，词汇的组合是无穷多的，而且有些组合并不存在于我们的语料库中，模型无法学习到这些词组。

语义角度和语法角度

语法分析的结果常常被表示成树（语法树）

参考链接： [http://blog.csdn.net/mengmengz07/article/details/51348554](http://blog.csdn.net/mengmengz07/article/details/51348554)

###  simple RNN ###

下面是具体的模型介绍。输入两个孩子节点的向量，输出节点合并之后的向量，以及这两个孩子节点适于合并的得分。

![](http://img.blog.csdn.net/20160508232733969?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

> score = UTP中的U是什么？

在剖析句子时，相邻节点试探性地两两合并，保留得分最高的那个，参与到下一轮试探中去，直至最终生成整个句子的向量。

![](http://img.blog.csdn.net/20160508232750644?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

整棵树的最后得分是各个合并节点的分数之和。

目标函数沿用了最大化间隔的思路