import os

import keras.utils.image_dataset
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing import image
from keras.utils import image_utils
from keras.applications.regnet import preprocess_input, decode_predictions
from keras import backend as K

from models.EEGModels_tensorflow import MI_EEGNet
# import seaborn
from to_plot import plot_confusion_matrix
import numpy as np
from load_datapro import BCI_IV_2a_mat, SW_BCI_IV_2a_mat
from loadmodel import Model
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

"""
    为已训练好的模型画 混淆矩阵 热力图
"""
def to_fenlei(fenlei):
    # 二分类任务
    #'769': left, '770': right, '771': foot, '772': tongue
    #'769': 1  ,  '770': 2  ,   '771': 3  ,  '772': 4
    # KIND = []
    if fenlei==0:
        KIND = ['left', 'right']
    elif fenlei==1:
        KIND = ['left', 'foot']
    elif fenlei==2:
        KIND = ['left', 'tongue']
    elif fenlei==3:
        KIND = ['right', 'foot']
    elif fenlei==4:
        KIND = ['right', 'tongue']
    elif fenlei==5:
        KIND = ['foot', 'tongue']
    else:
        print("you must to set the class")
    return KIND

def kappa(y_true,y_pred):
    """计算kappa值"""
    # print(y_true.shape,y_pred.shape)
    # print(y_true, "--------\n", y_pred)
    y_true = np.argmax(y_true, axis=1)
    # print(y_true, "--------\n", y_pred)
    # confusion_mat = confusion_matrix(y_true, y_pred)
    # conf_fig = seaborn.heatmap(confusion_mat, annot=True, fmt="d", cmap="BuPu")
    kappa = cohen_kappa_score(y_true, y_pred)
    # 保留四位小数
    kappa = round(kappa, 4)
    all_kappa.append(kappa)
    # print("Kappa value:", kappa)
    # return kappa


def kplot(class_id, subject_id, nb_classes, modelnum = 8, resamplie=True):
    # 加载数据
    (train_x, train_y),(test_x, test_y) = BCI_IV_2a_mat(class_id, subject_id, nb_classes=nb_classes,standardize=2,resamplie=resamplie)
    _, Chans, Samples, _ = train_x.shape
    # from sklearn.preprocessing import OneHotEncoder
    from keras import utils as np_utils
    # one-hot
    test_y = np_utils.to_categorical(test_y - 1)
    names = []
    #导入模型
    model, modelname = Model(modelnum=modelnum,nb_classes=nb_classes,Chans=Chans,Samples=Samples)
    model2, modelname2 = Model(modelnum=modelnum, nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    # print(model.summary())

    # l_r   l_f  l_t  r_f  r_t  f_t 二分类的分类标志
    makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
    # 模型名
    check_filebest = "best_model_A{:02d}.h5".format(subject_id)
    check_filefinal = "best_model_final_A{:02d}.h5".format(subject_id)
    # 模型路径
    if nb_classes == 2:
        # 二分类混淆矩阵Kind标签
        names = to_fenlei(class_id)
        checkpointer_best = os.path.join(
            os.getcwd() + "/kcheckpoint/" + modelname + '/classnum%d/' % (nb_classes) + makers[class_id],
            check_filebest)
        checkpointer_filefinal = os.path.join(
            os.getcwd() + "/kcheckpoint/" + modelname + '/classnum%d/' % (nb_classes) + makers[class_id],
            check_filefinal)
    elif nb_classes == 4:
        names = ["left", "right", "foot", "tongue"]
        checkpointer_best = os.path.join(os.getcwd() + "/kcheckpoint/" + modelname + '/classnum%d' % (nb_classes),
                                         check_filebest)
        checkpointer_filefinal = os.path.join(
                                    os.getcwd() + "/kcheckpoint/" + modelname + '/classnum%d/' % (nb_classes),
                                    check_filefinal)

    # load_weights best model
    model.load_weights(checkpointer_best)
    probs = model.predict(test_x)
    preds = probs.argmax(axis=-1)
    test_acc = round((100 * np.mean(preds == test_y.argmax(axis=-1))), 2)
    all_acc.append(test_acc)
    # print(names)
    # print("%s Classification accuracy: %f " % (check_filebest, test_acc))

    # 用于绘制混淆矩阵 left   right   foot   tongue
    # 1. plot the confusion matrices for both classifiers
    plt.figure()
    plot_confusion_matrix(preds, test_y.argmax(axis=-1), names, title='MI-EEGNet,{0}'.format(nb_classes))
    plt.show(block=False)
    # 混淆矩阵热力图
    # plt.figure()
    # plot_confusion_maxtirx(test_y, probs, names)
    # plt.show(block=False)

    # 2. best final model 混淆矩阵
    model2.load_weights(checkpointer_filefinal)
    probs = model2.predict(test_x)
    preds = probs.argmax(axis=-1)
    test_acc = round((100 * np.mean(preds == test_y.argmax(axis=-1))), 2)
    all_acc2.append(test_acc)
    # print("%s Classification accuracy: %f " % (check_filefinal, test_acc))

    # 用于绘制混淆矩阵 left   right   foot   tongue
    # plot the confusion matrices for both classifiers
    plt.figure()
    plot_confusion_matrix(preds, test_y.argmax(axis=-1), names, title='MI-EEGNet,{0}'.format(nb_classes))
    plt.show(block=False)

    # 混淆矩阵热力图
    # plt.figure()
    # plot_confusion_maxtirx(test_y, probs, names)
    # plt.show(block=False)


def sw_kplot(class_id, subject_id, nb_classes, modelnum = 8, resamplie=True):
    # 加载数据
    (train_x, train_y),(test_x, test_y) = SW_BCI_IV_2a_mat(class_id, subject_id, nb_classes=nb_classes,standardize=2,resamplie=resamplie)
    _, Chans, Samples, _ = train_x.shape
    # from sklearn.preprocessing import OneHotEncoder
    from keras import utils as np_utils
    # 测试
    test_y = np_utils.to_categorical(test_y - 1)
    names = []
    #导入模型
    model, modelname = Model(modelnum=modelnum,nb_classes=nb_classes,Chans=Chans,Samples=Samples)
    model2, modelname2 = Model(modelnum=modelnum, nb_classes=nb_classes, Chans=Chans, Samples=Samples)
    # print(model.summary())


    # l_r   l_f  l_t  r_f  r_t  f_t 二分类的分类标志
    makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
    # 模型名
    check_filebest = "best_model_A{:02d}.h5".format(subject_id)
    check_filefinal = "best_model_final_A{:02d}.h5".format(subject_id)
    # 模型路径
    if nb_classes == 2:
        # 二分类混淆矩阵Kind标签
        names = to_fenlei(class_id)
        checkpointer_best = os.path.join(
            os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id],
            check_filebest)
        checkpointer_filefinal = os.path.join(
            os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id],
            check_filefinal)
    elif nb_classes == 4:
        names = ["left", "right", "foot", "tongue"]
        checkpointer_best = os.path.join(os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d' % (nb_classes),
                                         check_filebest)
        checkpointer_filefinal = os.path.join(
            os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes),
            check_filefinal)

    # before predict
    # probs_before = model.predict(test_x)
    # preds_before = probs_before.argmax(axis=-1)
    # load_weights best model
    model.load_weights(checkpointer_best)
    probs = model.predict(test_x)
    preds = probs.argmax(axis=-1)

    # # 训练前后准确率分布
    # plot_scatter(preds_before, preds)

    # Kappa 计算
    kappa(test_y, preds)
    # 预测精度
    test_acc = round((100 * np.mean(preds == test_y.argmax(axis=-1))), 2)
    all_acc.append(test_acc)
    # print(names)
    # print("%s Classification accuracy: %f " % (check_filebest, test_acc))

    # 用于绘制混淆矩阵 left   right   foot   tongue
    # plot the confusion matrices for both classifiers
    plt.figure()
    plot_confusion_matrix(preds, test_y.argmax(axis=-1), names, title='MI-EEGNet,{0}'.format(nb_classes))
    plt.show(block=False)
    # 混淆矩阵热力图
    # plt.figure()
    # plot_confusion_maxtirx(test_y, probs, names)
    # plt.show(block=False)

    # best final model 混淆矩阵
    model2.load_weights(checkpointer_filefinal)
    probs = model2.predict(test_x)
    preds = probs.argmax(axis=-1)
    # Kappa 计算
    kappa(test_y, preds)

    test_acc = round((100 * np.mean(preds == test_y.argmax(axis=-1))), 2)
    all_acc2.append(test_acc)
    # 预测精度
    # print("%s Classification accuracy: %f " % (check_filefinal, test_acc))

    # 用于绘制混淆矩阵 left   right   foot   tongue
    # plot the confusion matrices for both classifiers
    plt.figure()
    plot_confusion_matrix(preds, test_y.argmax(axis=-1), names, title='MI-EEGNet,{0}'.format(nb_classes))
    plt.show(block=False)
    # 混淆矩阵热力图
    # plt.figure()
    # plot_confusion_maxtirx(test_y, probs, names)
    # plt.show(block=False)
def plot_scatter(accs_before, accs):
    # 创建一个新的图形
    fig, ax = plt.subplots()
    # 绘制散点图，x轴为分类前的准确率，y轴为分类后的准确率
    ax.scatter(accs_before, accs)
    # 添加坐标轴标签和标题
    ax.set_xlabel('Accuracy Before Classification')
    ax.set_ylabel('Accuracy After Classification')
    ax.set_title('Accuracy Constellation Map')
    # 显示图形
    plt.show()




if __name__ == '__main__':
    # 参数
    nb_classes = 4
    global all_acc, all_kappa, all_acc2
    all_acc = []
    all_acc2 = []
    all_kappa = []
    con = 1
    if nb_classes == 2:
        con = 6
    subject_ids = [7]
    print("开始%d分类混淆矩阵绘制......" % (nb_classes))
    for class_id in range(con):
        for subject_id in subject_ids:  # range(1, 10)
            # plot(class_id, subject_id, nb_classes=nb_classes, modelnum=8, resamplie=True)
            # sw_plot(class_id, subject_id, nb_classes=nb_classes, modelnum=8, resamplie=True)

            # kplot(class_id, subject_id, nb_classes=nb_classes, modelnum=8, resamplie=True)
            sw_kplot(class_id, subject_id, nb_classes=nb_classes, modelnum=8, resamplie=True)
    if all_acc:
        print("best_model_ALL_ACC:", all_acc)
    if all_acc2:
        print("final_model_ALL_ACC:", all_acc2)
    if all_kappa:
        print("SW_ALL_Kappa:", all_kappa)
    plt.show()