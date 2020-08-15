import util1
import numpy as np
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo_feature, max_length = 40):
    """
    根据输入的图像特征产生图像的标题
    :param model: 预先训练好的图像标题生成神经网络模型
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征, 为VGG16网络修改版产生的特征
    :param max_length: 训练数据中最长的图像标题的长度
    :return: 产生的图像的标题
    """


    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence,maxlen = max_length)
        output = model.predict([photo_feature,sequence])
        integer = np.argmax(output)
        word = word_for_id(integer,tokenizer)
        if word is None:
            break
        in_text = in_text + " " + word
        if word == "endseq":
            break
    return in_text

# TODO：如何评价模型
def evaluate_model(model, captions, photo_features, tokenizer, max_length = 40):
    """计算训练好的神经网络产生的标题的质量,根据4个BLEU分数来评估

    Args:
        model:　训练好的产生标题的神经网络
        captions: dict, 测试数据集, key为文件名(不带.jpg后缀), value为图像标题list
        photo_features: dict, key为文件名(不带.jpg后缀), value为图像特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_length：训练集中的标题的最大长度

    Returns:
        tuple:
            第一个元素为权重为(1.0, 0, 0, 0)的ＢＬＥＵ分数
            第二个元素为权重为(0.5, 0.5, 0, 0)的ＢＬＥＵ分数
            第三个元素为权重为(0.3, 0.3, 0.3, 0)的ＢＬＥＵ分数
            第四个元素为权重为(0.25, 0.25, 0.25, 0.25)的ＢＬＥＵ分数

    """
    actual, predicted = list(),list()
    for key, caption_list in captions.items():
        yhat = generate_caption(model,tokenizer,photo_features[key],max_length)
        references = [d.split() for d in caption_list]
        actual.append(references)
        predicted.append(yhat.split())
    # print(actual)
    # print(predicted)
    # print(len(actual))  # 1000
    # print(len(predicted))  # 1000
    blue1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    blue2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    blue3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.4, 0))
    blue4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    print('BLEU-1: %f' % blue1)
    print('BLEU-2: %f' % blue2)
    print('BLEU-3: %f' % blue3)
    print('BLEU-4: %f' % blue4)
    return blue1,blue2,blue3,blue4


if __name__ == '__main__':

    # # TODO: 测试某一张图片产生的文字
    # #  如果要进行测试的话，应该要拿到测试图片的feature，所有图片的feature已经存放在features.pkl里面，
    # #  所以只需修改读取测试图片的TXT文件即可
    #
    # # 加载已经训练好的模型
    # model_path = r"E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/model_0.h5"
    # model = load_model(model_path)
    #
    # # 读取测试图片TXT文件
    # filename = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/Flickr_8k.testImages.txt'
    # train = util1.load_ids(filename)  # 返回了一个{}，包含了文件名（去除.jpg）
    #
    # # 读取描述文件
    # des_path = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task5/descriptions.txt'
    # train_captions = util1.load_clean_captions(des_path, train)
    #
    # feature_path = r"E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/features.pkl"
    # train_features = util1.load_photo_features(feature_path, train)
    #
    # tokenizer = load(open(r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/tokenizer.pkl', 'rb'))
    # vocab_size = len(tokenizer.word_index) + 1  # 词汇表大小
    # max_len = util1.get_max_length(train_captions)
    #
    # text = generate_caption(model,tokenizer, train_features["2677656448_6b7e7702af"], max_length = 40)
    # print(text)

    #TODO： evaluate_model(model, captions, photo_features, tokenizer, max_length=40)
    model_path = r"E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/model_9.h5"
    model = load_model(model_path)

    # 读取测试图片TXT文件
    filename = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/Flickr_8k.testImages.txt'
    test = util1.load_ids(filename)  # 返回了一个{}，包含了文件名（去除.jpg）
    des_path = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task5/descriptions.txt'
    captions = util1.load_clean_captions(des_path, test)

    feature_path = r"E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/features.pkl"
    photo_features = util1.load_photo_features(feature_path, test)

    tokenizer = load(open(r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/homework-master-7fc833414b95225130c323c278230bc388af5c6b/homework1/task4/tokenizer.pkl','rb'))
    max_length = util1.get_max_length(captions)
    blue1,blue2,blue3,blue4 = evaluate_model(model, captions, photo_features, tokenizer, max_length=40)
    # BLEU-1: 0.563778
    # BLEU-2: 0.296756  结果不好是因为模型训练得不够好，只训练了一次模型？
    # BLEU-3: 0.124940
    # BLEU-4: 0.069551
