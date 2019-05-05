# -*- coding:utf-8 -*-
# @author: Vince1359
# @file: model.py
# @time: 2019/4/22 12:23

from tensorflow.python import keras


def DarknetConv2D_BN_Leaky(x, filters, kernel_size, strides=(1, 1), padding='same', use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(5e-4)):
    """
    实现DBL组件，即一个Conv2D -> BatchNormalization -> LeakyReLU组合，该组合在YOLOv3中是基本组件
    :param x: 输入层
    :param filters: 卷积层输出通道数
    :param kernel_size: 卷积核大小
    :param strides: 卷积核移动步长
    :param padding: 补齐方式，'same'或'valid'
    :param use_bias: 布尔值，是否使用偏置向量
    :param kernel_regularizer: 对卷积核使用的正则化项，默认为L2正则化
    :return x: 输出层
    """
    x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                            use_bias=use_bias, kernel_regularizer=kernel_regularizer)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


def DBLs_Stack(x, filters):
    """
    五个DBL的组合
    :param x: 输入层
    :param filters: 卷积层输出通道数
    :return x: 输出层
    """
    x = DarknetConv2D_BN_Leaky(x, filters, (1, 1))
    x = DarknetConv2D_BN_Leaky(x, filters * 2, (3, 3))
    x = DarknetConv2D_BN_Leaky(x, filters, (1, 1))
    x = DarknetConv2D_BN_Leaky(x, filters * 2, (3, 3))
    x = DarknetConv2D_BN_Leaky(x, filters, (1, 1))
    return x


def Res_Unit(x, filters):
    """
    实现残差网络基本组件，即两个DBL组件结合一个残差相加结构
    :param x: 输入层
    :param filters: 第二个卷积层输出通道数，第一个卷积层为该数量的一半
    :return x: 输出层
    """
    y = DarknetConv2D_BN_Leaky(x, filters//2, (1, 1))
    y = DarknetConv2D_BN_Leaky(y, filters, (3, 3))
    x = keras.layers.Add()([x, y])
    return x


def Resblock_Body(x, filters, blocks):
    """
    残差网络块基本组件，即ZeroPadding2D -> DBL -> blocks个Res_Unit
    :param x: 输入层
    :param filters: 输出通道数
    :param blocks: Res_Unit堆叠个数
    :return: 输出层
    """
    x = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(x, filters, (3, 3), strides=(2, 2), padding='valid')
    for i in range(blocks):
        x = Res_Unit(x, filters)
    return x


def YOLO_v3(inputs, out_dim):
    """
    生成YOLOv3模型
    :param inputs: 输入层
    :param out_dim: 输出通道数，等于每个小格预测bonding_box个数 * (类别数量 + 5)，此处为3*(80+5)=255
    :return model: YOLOv3模型
    """
    x = DarknetConv2D_BN_Leaky(inputs, 32, (3, 3))
    x = Resblock_Body(x, 64, 1)
    x = Resblock_Body(x, 128, 2)

    # y3支路起点x3，输出为13*13*255
    x3 = Resblock_Body(x, 256, 8)

    # y2支路起点x2，输出为26*26*255
    x2 = Resblock_Body(x3, 512, 8)

    # y1支路起点x1，输出为52*52*255
    x1 = Resblock_Body(x2, 1024, 4)

    # y1支路经过5个DBL之后的结果，分为2支，一支从y1进入y2，另一支留在y1
    x12 = DBLs_Stack(x1, 512)

    # y1支路
    x1 = DarknetConv2D_BN_Leaky(x12, 1024, (3, 3))
    y1 = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                             kernel_regularizer=keras.regularizers.l2(5e-4))(x1)

    # x12分路经过DBL，上采样，再跟x2分路合并为y2分支
    x12 = DarknetConv2D_BN_Leaky(x12, 256, (1, 1))
    x12 = keras.layers.UpSampling2D(2)(x12)
    x2 = keras.layers.Concatenate()([x2, x12])

    # y2支路经过5个DBL周后的结果，分为2支，一支从y2进入y3，另一支留在y2
    x23 = DBLs_Stack(x2, 256)

    # y2支路
    x2 = DarknetConv2D_BN_Leaky(x23, 512, (3, 3))
    y2 = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                             kernel_regularizer=keras.regularizers.l2(5e-4))(x2)

    # x23分路经过DBL，上采样，再跟x3分路合并为y3分支
    x23 = DarknetConv2D_BN_Leaky(x23, 128, (1, 1))
    x23 = keras.layers.UpSampling2D(2)(x23)
    x3 = keras.layers.Concatenate()([x3, x23])

    # y3支路
    x3 = DBLs_Stack(x3, 128)
    x3 = DarknetConv2D_BN_Leaky(x3, 256, (3, 3))
    y3 = keras.layers.Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same', use_bias=False,
                             kernel_regularizer=keras.regularizers.l2(5e-4))(x3)

    return keras.models.Model(inputs, [y1, y2, y3])


if __name__ == '__main__':
    my_model = YOLO_v3(keras.Input(shape=(416, 416, 3), name='image'), 255)
    print(my_model.summary())
