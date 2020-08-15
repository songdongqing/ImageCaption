from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras


def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    # 从json文件中加载模型
    with open(r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/网络/vgg16_exported.json', 'r') as file:
        model_json = file.read()

    model = model_from_json(model_json)
    model.load_weights(r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/网络/vgg16_exported.h5')

    return model

def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    # 'RGB'->'BGR'(Matplotlib image to OpenCV):
    # https://www.scivision.co/numpy-image-bgr-to-rgb/

    x = x[..., ::-1]  # 将数组里面的每个数 反序
    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.

    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    img = pil_image.open(path)
    img = img.resize(target_size, pil_image.NEAREST)
    return np.asarray(img, dtype=K.floatx())   # (224, 224, 3)

def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]
       输出的维度为4096维

    Args:
        directory: 包含jpg文件的文件夹

    Returns:
        None
    """
    model = load_vgg16_model()
    # 去除模型的最后一层
    # resnet_model = Model(inputs=base_model.input, outputs=base_model.get_layer('max_pooling2d_6').output)
    # #'max_pooling2d_6'其实就是上述网络中全连接层的前面一层，当然这里你也可以选取其它层，把该层的名称代替'max_pooling2d_6'即可，
    # 这样其实就是截取网络，输出网络结构就是方便读取每层的名字。

    # model.layers.pop()
    # model = Model(inputs=model.inputs,outputs = model.layers[-1].output)
    model = Model(inputs=model.inputs, outputs=model.get_layer("fc2").output)

    features = dict()
    for fn in listdir(directory):
        filename = directory + "/" + fn
        arr = load_img_as_np_array(filename,target_size=(224,224))
        # 改变数组的形态，增加一个维度（批处理输入的维度）
        arr = arr.reshape((1,arr.shape[0],arr.shape[1],arr.shape[2]))
        # 预处理图像作为VGG模型的输入
        arr = preprocess_input(arr)
        # 计算特征
        feature = model.predict(arr,verbose=0)
        # id = fn 去掉文件后缀
        id = fn.split(".")[0]
        features[id] = feature
    return features



if __name__ == '__main__':
    # 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    directory = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/Flicker8k'
    # 测试用的，只有10张图片
    # directory = r'E:/AI资源计算机视觉/JM07 - TXXY - CV2期/02.资料/copyFli'
    features = extract_features(directory)
    # print(len(features["10815824_2997e03d76"][0]))  # 4096
    print('提取特征的文件个数：%d' % len(features))  # 8091
    print(keras.backend.image_data_format())  # 返回默认的图像维度顺序  channels_last
    #保存特征到文件
    dump(features, open('features.pkl', 'wb'))





