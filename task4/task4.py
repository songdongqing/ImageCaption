from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from pickle import load
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from numpy import array
import util  # 貌似这样也能运行
import spacy
import numpy as np


def create_batches(desc_list, photo_features, tokenizer, max_len, vocab_size=7378):
    """从输入的图片标题list和图片特征构造LSTM的一组输入

    Args:
        desc_list: 某一个图像对应的一组标题(一个list)
        photo_features: 某一个图像对应的特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_len: 训练数据集中最长的标题的长度
        vocab_size: 训练集中的单词个数, 默认为7378

    Returns:
        tuple:
            第一个元素为list, list的元素为图像的特征
            第二个元素为list, list的元素为图像标题的前缀
            第三个元素为list, list的元素为图像标题的下一个单词(根据图像特征和标题的前缀产生)

    Examples:
        #>>> from pickle import load
        #>>> tokenizer = load(open('tokenizer.pkl', 'rb'))
        #>>> desc_list = ['startseq one dog on desk endseq', "startseq red bird on tree endseq"]
        #>>> photo_features = [0.434, 0.534, 0.212, 0.98]
        #>>> print(create_batches(desc_list, photo_features, tokenizer, 6, 7378))
            (array([[ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ]]),
            array([[   0,    0,    0,    0,    0,    2],
                   [   0,    0,    0,    0,    2,   59],
                   [   0,    0,    0,    2,   59,    9],
                   [   0,    0,    2,   59,    9,    6],
                   [   0,    2,   59,    9,    6, 1545],
                   [   0,    0,    0,    0,    0,    2],
                   [   0,    0,    0,    0,    2,   26],
                   [   0,    0,    0,    2,   26,  254],
                   [   0,    0,    2,   26,  254,    6],
                   [   0,    2,   26,  254,    6,  134]]),
            array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   ...,
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))

    """
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:    # desc为图片的标题
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(photo_features)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# data generator, intended to be used in a call to model.fit_generator()
def data_generator(captions, photo_features, tokenizer, max_len):
    """创建一个训练数据生成器, 用于传入模型训练函数的第一个参数model.fit_generator(generator,...)

    Args:
        captions: dict, key为图像名(不包含.jpg后缀), value为list, 图像的几个训练标题
        photo_features: dict, key为图像名(不包含.jpg后缀), value为图像的特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_len: 训练集中的标题最长长度

    Returns:
        generator, 使用yield [[list, 元素为图像特征, list, 元素为输入的图像标题前缀], list, 元素为预期的输出图像标题的下一个单词]

    """
    # loop for ever over images
    while 1:
        for key, desc_list in captions.items():
            # retrieve the photo feature
            photo_feature = photo_features[key][0]  # [],图片特征，4096维
            in_img, in_seq, out_word = create_batches(desc_list, photo_feature, tokenizer, max_len)
            # yield [[in_img, in_seq], out_word]
            yield ([in_img, in_seq], out_word)


    # photo_feature = photo_features["3227594168_3351722aae"][0]
    # print(photo_feature)  # [0.62417436 0.         0.         ... 0.         0.         0.        ]
    # print(len(photo_feature))  # 4096
    # desc_list = captions["3227594168_3351722aae"]
    # print(desc_list)  # [],5个元素，分别为图片特征对应的5个答案
    # in_img, in_seq, out_word = create_batches(desc_list, photo_feature, tokenizer, max_len)
    # print("开始转为向量了。。。。")
    # print(in_img)
    # print(len(in_img))  # 69
    # print(in_img[0])
    # print(len(in_img[0]))  # 4096个特征
    # print("----上面是图片特征-------")
    # print(in_seq)
    # print(len(in_seq))  # 69
    # print(in_seq[0])
    # print(len(in_seq[0]))  # 40维，最大输入句子数
    # print("----上面是输入特征-------")
    # print(out_word)
    # print(len(out_word))  # 69
    # print(out_word[0])
    # print(len(out_word[0]))  # 7378
    # print("----上面是对应得answer特征-------")


def caption_model(vocab_size, max_len):
    """创建一个新的用于给图片生成标题的网络模型，合并之前的两个模型

    Args:
        vocab_size: 训练集中标题单词个数
        max_len: 训练集中的标题最长长度

    Returns:
        用于给图像生成标题的网络模型

    """
    input_1 = Input(shape=(4096,))
    droput_1 = Dropout(0.5)(input_1)
    dense_1 = Dense(256, activation='relu')(droput_1)

    input_2 = Input(shape=(max_len,))
    embedding_1 = Embedding(vocab_size, 256)(input_2)
    droput_2 = Dropout(0.5)(embedding_1)
    lstm_1 = LSTM(256)(droput_2)

    add_1 = add([dense_1, lstm_1])
    dense_2 = Dense(256, activation='relu')(add_1)
    outputs = Dense(vocab_size, activation='softmax')(dense_2)

    model = Model(inputs=[input_1, input_2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def train():
    # load training dataset (6K)
    filename = 'Flickr_8k.trainImages.txt'
    train = util.load_ids(filename)  # 返回了一个{元素，元素，元素......}，包含了文件名（去除.jpg）
    print('Dataset: %d' % len(train))  # 6000
    train_captions = util.load_clean_captions('descriptions.txt', train)
    print('Captions: train number=%d' % len(train_captions))
    # photo features
    train_features = util.load_photo_features('features.pkl', train)
    print('Photos: train=%d' % len(train_features))
    # prepare tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    # determine the maximum sequence length
    max_len = util.get_max_length(train_captions)
    print('Description Length: %d' % max_len)

    # define the model
    # TODO：创建model
    model = caption_model(vocab_size, max_len)
    # train the model, run epochs manually and save after each epoch
    epochs = 20
    steps = len(train_captions)
    for i in range(epochs):
        # create the data generator
        generator = data_generator(train_captions, train_features, tokenizer, max_len)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model.save('model_' + str(i) + '.h5')

def add_weights():
    # create word embeddings
    # spacy.cli.download("en_core_web_lg")
    en_path = r""
    nlp = spacy.load(en_path)
    print(nlp)

    # # create word embeddings
    # embedding_dimension = 300
    # embedding_matrix = np.zeros((vocab_len, embedding_dimension))
    #
    # # travel through every word in vocabulary and get its corresponding vector
    # for word, index in tokenizer.word_index.items():
    #     doc = nlp(word)
    #     embedding_vector = np.array(doc.vector)
    #     embedding_matrix[index] = embedding_vector
    #
    # # adding embeddings to model
    # predictive_model.layers[2]
    # predictive_model.layers[2].set_weights([embedding_matrix])
    # predictive_model.layers[2].trainable = False
    pass

if __name__ == '__main__':
    # add_weights()



    filename = 'Flickr_8k.trainImages.txt'
    train = util.load_ids(filename)  # 返回了一个{}，包含了文件名（去除.jpg）

    des_path = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task5/descriptions.txt'
    train_captions = util.load_clean_captions(des_path, train)
    # print('Captions: train number=%d' % len(train_captions))
    # print(train_captions["3227594168_3351722aae"])  # ['startseq Two blonde ladies wearing sunglasses lounge on the grass with a dacshund . endseq',
    # 'startseq Two blonde young women hang out in the grass with a brown dog . endseq',
    # 'startseq Two blond women sit in grass with a small dog . endseq',
    # 'startseq Two women laying on grass with a dog . endseq',
    # 'startseq Two women on a grassy hill lit by the sun ; one looks at a dachshund , the other looks to the side . endseq']

    # photo features
    feature_path = r"E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/features.pkl"
    train_features = util.load_photo_features(feature_path, train)
    # print('Photos: train=%d' % len(train_features))
    # print(len(train_features["3585117340_73e96b6173"][0]))  # 4096

    # prepare tokenizer
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小
    print('Vocabulary Size: %d' % vocab_size)  # 7378

    # determine the maximum sequence length
    max_len = util.get_max_length(train_captions)
    print('Description Length: %d' % max_len)  # 40

    # define the model
    model = caption_model(vocab_size, max_len)
    print(model)
    # train the model, run epochs manually and save after each epoch
    epochs = 20
    steps = len(train_captions)  # 6000
    for i in range(epochs):
        # create the data generator
        generator = data_generator(train_captions, train_features, tokenizer, max_len)
        # print(generator)
        # fit for one epoch
        model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
        # save model
        model.save('model_' + str(i) + '.h5')

# ValueError: No gradients provided for any variable: ['embedding/embeddings:0', 'dense/kernel:0'  生成器应该返回的是元组