### 任务一：使用keras创建VGG16定义的cnn网络结构
* 作业中已经给出部分框架，需要根据任务一的图片中VGG16的结构完善网络结构。

### 任务二：将flicker8k的图像文件转为图像特征，保存为字典pickle文件
* 从给定的VGG16网络结构文件和权值文件，创建VGG16网络。
* 修改网络结构(去除最后一层)。
* 利用修改的网络结构，提取flicker8k数据集中所有图像特征，使用字典存储，key为文件名(不带.jpg后缀)，value为一个网络的输出。
* 将字典保存为features.pkl文件(使用pickle库)。

### 任务三：完成create_input_data_for_one_image函数
* 为了训练LSTM，训练数据中的每一个图像的每一个标题都需要被重新拆分为输入和输出部分。如果标题为“a cat sits on the table”，需要添加起始和结束标志，变为“startseq a cat sits on the table endseq”，再从它产生训练数据序列。
* 另外，需要预处理单词，去掉's和一些不需要的标点符号，还需要将每一个单词转换为一个整数。

### 任务四：构建自动产生图像标题的网络结构
* 可以按照所给网络结构的思路构建，然后训练网络。
* 注意LSTM的第一层应该是一个嵌入层(embedding layer)，用于将整数表达的单词转换为向量表达。
* 使用交叉验证cross_validation来衡量不同的结构的优劣。

### 任务五：完成预测generate_caption代码，了解如何评价模型的性能
* 使用4个corpus BLEU分数来评价模型在测试集上面的表现
* 可以根据这个评价回头去修改你的网络，可以重新训练

### 提交方式
* 将作业打包发送至 ghostlyfeng@outlook.com 邮箱（命名：姓名-homework1/2/3...（和Gitlab上作业一 一对应））

>>>>>>> 
