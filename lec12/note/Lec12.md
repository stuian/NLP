## 解读论文：Memory Networks ##

中文博客参考链接：

1、[http://m.blog.csdn.net/wang735019/article/details/53909079](http://m.blog.csdn.net/wang735019/article/details/53909079)

2、[http://blog.csdn.net/u011274209/article/details/53384232?ref=myread](http://blog.csdn.net/u011274209/article/details/53384232?ref=myread)

memory networks有两部分，inference components和long-term memory components(在别人的基础上新增的:可以被用来读取和写入)

论文中：

- 第二部分介绍了memory networks的整体框架；
- 第三部分展示了一个文本领域的问答任务的确切应用；
- 第四部分讨论了相关的其他论文；
- 第五部分描述了我们的试验；第六部分最后进行总结。

### 一、整体框架 ###

1、a memory m(an array of objects indexed by mi)

> m = [m1
     m2
	 ...
	 mi]
	(mi是向量)

2、four (potentially learned) components I, G, O and R as follows:

- I: (input feature map) – converts the incoming input to the internal feature representation.（输入将来映射-把输入转换成内部特征表征）
- G: (generalization) – updates old memories given the new input. We call this generalization as there is an opportunity for the network to compress and generalize its memories at this stage for some intended future use.（整体化-根据新的输入更新旧的记忆，我们把这种整体化看作是网络的一个机会，用来在这个阶段压缩和合并它的记忆，以便将来作为特定的使用）
- (output feature map) – produces a new output (in the feature representation space), given the new input and the current memory state.（在给定新的输入和现在的记忆状态下，产生一个新的输入（在将来的表征空间里））
- (response) – converts the output into the response format desired. For example, a textual
response or an action.（把输出转换成希望回复的形式。例如，一个文本的回复或者行为）

Given an input x (e.g., an input character, word or sentence depending on the granularity chosen, an image or an audio signal) the flow of the model is as follows:

- 1. Convert x to an internal feature representation I(x).
- 2. Update memoriesmi given the new input: mi = G(mi, I(x),m), 8i.
- 3. Compute output features o given the new input and the memory: o = O(I(x),m).
- 4. Finally, decode output features o to give the final response: r = R(o).

G（补充）:

![](https://github.com/stuian/NLP/blob/master/lec12/images/1.jpg?raw=true)

I(x)存储在记忆m中的一个位置，H(x)是一个位置选取函数（slot choosing functionH），用来指定I(x)在记忆中的位置。

- 如果输入的数据是字符级别或者字词级别的，可以把这些数据组合成一个数据块
- 如果输入的数据太大，则可以通过H(X)函数把这些数据按照一定的主题归类，同时，不一定要运行所有的数据,取需要的相关主题的数据即可。
- 如果记忆“满了”，则需要(通过H函数)忘记一些记忆。H可以对每一条记忆的使用性进行评分。

O：

读取记忆，推断出对回复有关的记忆。

R：

可以是一个建立在O的输出基础上的RNN。

![](http://img.blog.csdn.net/20161229094352510?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcWZudV9janRfd2w=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 二、A MEMNN IMPLEMENTATION FOR TEXT ###

> MEMNN是本文介绍的模型

**figure1（例子）：**

 - Joe went to the kitchen. Fred went to the kitchen. Joe picked up the milk.
-  Joe travelled to the office. Joe left the milk. Joe went to the bathroom.
-  Where is the milk now? A: office
-  Where is Joe? A: bathroom
-  Where was Joe before the office? A: kitchen

>　回答牛奶的位置需要理解ｐｉｃｋ　ｕｐ和ｌｅｆｔ这个两个行为；要想回答ｊｏｅ在办公室之前在哪就必须理解时间因素

２．１　BASIC MODEL

假设输入（ｉｎｐｕｔ　ｔｅｘｔ）的是句子（不管是故事的陈述还是问题）

> using an embedding model to represent text

- S(x) returns the next empty memory slot N:![](https://github.com/stuian/NLP/blob/master/lec12/images/2.jpg?raw=true)
- The G module is thus only used to store this new memory, so old memories are not updated.
- The core of inference lies in the O and R modules.

通过输入，O模块产生输出特征，找出K个supporting memory。

k = 1,分数最高的支撑记忆通过下列公式计算：

![](https://github.com/stuian/NLP/blob/master/lec12/images/3.jpg?raw=true)

> sO is a function that scores the match between the pair of sentences x and mi.

k = 2

![](https://github.com/stuian/NLP/blob/master/lec12/images/4.jpg?raw=true)

#### response: ####

![](https://github.com/stuian/NLP-CS224d/blob/master/lec12/images/1.png?raw=true)

整个模型过程的关键在函数s(x,y) 的实现。

> so(x,y)和sr(x,y)是一样的

其中参数U的维度为nXD , D代表特征的数量，n代表嵌入维度。Φx和Φy 将原始的输入文本映射到D维的特征空间。文章中D=3|W| ，W是词表大小。
![](http://img.blog.csdn.net/20161230213726715?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcWZudV9janRfd2w=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
词典中的每个词都有3种不同的表示方式，其中一份给予Φy，另两份给Φx。若输入x是问题中的词，则映射在位置1，若输入词是“记忆支撑”，那么映射在位置2，y映射在位置3。若其中某个位置未映射则置0。 


### 训练 ###

模型的训练是一个监督学习的过程，使用最大距离损失和梯度下降法训练参数。

[http://img.blog.csdn.net/20161230215314460?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcWZudV9janRfd2w=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast](http://img.blog.csdn.net/20161230215314460?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcWZudV9janRfd2w=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
