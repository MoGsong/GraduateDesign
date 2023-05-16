
import numpy as np
import os.path

from load_datapro import SW_BCI_IV_2a_mat
import tensorflow as tf
from keras import backend as K
from loadmodel import Model
import keras.optimizers



def Kfoldtrain(class_id=1,subject_id=3,nb_classes=2,modelnum=8,opt=1,kfoldnum=5,epochs=500,Es=True,patience=100,extra_train=False,extra_epoch=50,resamplie=True):

    K.set_image_data_format('channels_last')

    info = {'n_epochs_kfold': [], 'best_model': [], 'fold_accuracy_train': [], 'fold_accuracy_val': [],
            'test_accuracy_before': [], 'test_accuracy_after': [],
            'train_loss_before':[],'val_loss_before':[],
            'train_loss_after':[],'val_loss_after':[],'train_acc_after':[],'val_acc_after':[]}
    #   fenlei:
    #   1：l_r   2:l_f  3:l_t  4:r_f   5:r_t   6:f_t

    # 导入训练数据
    (train_x, train_y),(test_x, test_y) = SW_BCI_IV_2a_mat(class_id, subject_id, nb_classes=nb_classes,standardize=2,resamplie=resamplie)
    _, Chans, Samples, _ = train_x.shape

    # 打乱数据顺序
    ziped = list(zip(train_x, train_y))
    np.random.shuffle(ziped)
    train_x, train_y = zip(*ziped)
    train_x, train_y = np.array(train_x), np.array(train_y)
    # 测试集
    ziped = list(zip(test_x, test_y))
    np.random.shuffle(ziped)
    test_x, test_y = zip(*ziped)
    test_x,test_y = np.array(test_x), np.array(test_y)

    # from sklearn.preprocessing import OneHotEncoder
    from keras import utils as np_utils
    # train_y = np_utils.to_categorical(train_y - 1)
    test_y = np_utils.to_categorical(test_y - 1)

    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder()
    # enc.fit(train_y)
    # train_y = enc.transform(train_y).toarray()
    # test_y = enc.transform(test_y).toarray()

    # Training folds
    All_model = []
    All_AccuracyTrain = []
    All_AccuracyVal = []
    All_AccuracyTest = []
    All_loss = []
    All_epochs = []

    # kfoldnum = 5
    # epochs = 500
    # extra_epoch = 50

    # foldtrain_loss = np.zeros([kfoldnum,epochs])
    # foldval_loss = np.zeros([kfoldnum,epochs])

    from sklearn.model_selection import StratifiedKFold     # KFold,
    kfold = StratifiedKFold(n_splits=kfoldnum, shuffle=True)
    fold = 1
    while fold<=kfoldnum:
        #导入模型
        global model
        model,modelname = Model(modelnum=modelnum,nb_classes=nb_classes,Chans=Chans,Samples=Samples)
        print('----------------------fold-%d-%s----------------------'%(fold,modelname))
        if opt==1:
            # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        elif opt ==2:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['categorical_accuracy'])
        # print("-----------------------train_x--------------------------", train_x.shape, train_x)
        # print("-----------------------train_y--------------------------", train_y.shape, train_y)
        train, val = list(kfold.split(train_x, train_y))[fold - 1]   # split(train_x, train_y)
        X = train_x[train,]
        Y = train_y[train,]
        X_val = train_x[val,]
        Y_val = train_y[val,]
        # 合并验证集和训练集
        # X = np.concatenate([X, X_val], axis=0)
        # Y = np.concatenate([Y, Y_val], axis=0)

        Y = np_utils.to_categorical(Y - 1)
        Y_val = np_utils.to_categorical(Y_val - 1)


        from keras.callbacks import ModelCheckpoint, EarlyStopping

        # l_r   l_f  l_t  r_f  r_t  f_t
        makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
        check_file = "best_model_A{:02d}.h5".format(subject_id)
        check_filefinal = "best_model_final_A{:02d}.h5".format(subject_id)
        if nb_classes == 2:
            checkpointer_file = os.path.join(
                    os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id], check_file)
            checkpointer_filefinal = os.path.join(
                    os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id], check_filefinal)
            file_name = os.path.join(
                    os.getcwd() + "/info/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id], "info_A{:02d}.mat".format(subject_id))
            if not os.path.exists(checkpointer_file):
                os.makedirs(os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id],
                            exist_ok=True)
                os.makedirs(os.getcwd() + "/info/" + modelname + '/SlideWindow_class%d/' % (nb_classes) + makers[class_id],
                            exist_ok=True)
            else:
                model.load_weights(checkpointer_file)

        elif nb_classes == 4:
            checkpointer_file = os.path.join(
                    os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d' % (nb_classes),check_file)
            checkpointer_filefinal = os.path.join(
                    os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d' % (nb_classes),check_filefinal)
            file_name = os.path.join(
                    os.getcwd() + "/info/" + modelname + '/SlideWindow_class%d/' % (nb_classes) , "info_A{:02d}.mat".format(subject_id))
            if not os.path.exists(checkpointer_file):
                os.makedirs(os.getcwd() + "/kcheckpoint/" + modelname + '/SlideWindow_class%d/' % (nb_classes),
                            exist_ok=True)
                os.makedirs(os.getcwd() + "/info/" + modelname + '/SlideWindow_class%d/' % (nb_classes),
                            exist_ok=True)
            else:
                model.load_weights(checkpointer_file)

        # checkpointer = ModelCheckpoint(filepath=checkpointer_file, monitor='val_accuracy', mode='max', verbose=1,
        #                                save_best_only=True)
        checkpointer = ModelCheckpoint(filepath=checkpointer_file, verbose=0,
                                       save_best_only=True)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)                            # max -》min(val_acc), mode = "min" min(loss)
        if Es:
            callbacks = [checkpointer,es]
        else:
            callbacks = [checkpointer]
        fittedModel = model.fit(X, Y, batch_size = 64, epochs = epochs,
                                verbose = 2, validation_data=(X_val, Y_val),  # X_val, Y_val ;  test_x, test_y
                                callbacks=callbacks)

        val_loss = fittedModel.history['val_loss']
        train_loss = fittedModel.history['loss']


        All_loss.append(np.amin(val_loss))
        #计算模型的参数
        numParams   = model.count_params()
        print(numParams)

        if es.stopped_epoch == 0:
            All_epochs.append(epochs)
        else:
            All_epochs.append(es.stopped_epoch)

        model.load_weights(checkpointer_file)
        All_model.append(model)

        fold += 1

        probs = model.predict(X)
        preds = probs.argmax(axis=-1)
        train_acc = round(100 * np.mean(preds == Y.argmax(axis=-1)), 2)
        All_AccuracyTrain.append(train_acc)

        probs = model.predict(X_val)
        preds = probs.argmax(axis=-1)
        val_acc = round(100 * np.mean(preds == Y_val.argmax(axis=-1)), 2)

        All_AccuracyVal.append(val_acc)

        probs = model.predict(test_x)
        preds = probs.argmax(axis=-1)
        test_acc = round(100 * np.mean(preds == test_y.argmax(axis=-1)), 2)

        All_AccuracyTest.append(test_acc)



        # val_loss_before = foldval_loss[np.argmin(All_loss),:]
        # train_loss_before = foldtrain_loss[np.argmin(All_loss), :]
        # info['train_loss_before'].append(train_loss_before)
        # info['val_loss_before'].append(val_loss_before)

    All_model[np.argmin(All_loss)].save_weights(checkpointer_file)

    model,modelname= Model(modelnum=modelnum,nb_classes=nb_classes,Chans=Chans,Samples=Samples)
    print('------predict model : %s------' % (modelname))
    model.load_weights(checkpointer_file)
    probs = model.predict(test_x)
    preds = probs.argmax(axis=-1)
    test_acc = round(100 * np.mean(preds == test_y.argmax(axis=-1)), 2)
    test_acc_before = test_acc

    info['test_accuracy_before'].append(test_acc_before)

    # print('------------------A0%d------------------'%(subject_id))
    if extra_train:
        print('------extra_train   : %s------' % (modelname))
        from keras import utils as np_utils
        train_y = np_utils.to_categorical(train_y - 1)
        test_y = np_utils.to_categorical(test_y - 1)
        if opt==1:
            # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
        elif opt ==2:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        checkpointer = ModelCheckpoint(checkpointer_filefinal,
                                       monitor='val_accuracy', mode='max', verbose=0,save_best_only=True)
        fittedModel = model.fit(train_x, train_y, batch_size=64, epochs=extra_epoch, validation_freq=1,
                                verbose=2, validation_data=(test_x, test_y), callbacks=[checkpointer])
        val_loss_after = fittedModel.history['val_loss']
        train_loss_after = fittedModel.history['loss']
        val_acc_after = fittedModel.history['val_accuracy']     # val_categorical_accuracy # val_accuracy
        train_acc_after = fittedModel.history['accuracy']        # categorical_accuracy
        info['val_loss_after'].append(val_loss_after)
        info['train_loss_after'].append(train_loss_after)
        info['val_acc_after'].append(val_acc_after)
        info['train_acc_after'].append(train_acc_after)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(1, 2, 1)  # 将图像分为1行2列，这段代码，画出第1列
        # plt.plot(train_acc_after, label='Training Accuracy')  # 画出acc数据
        # plt.plot(val_acc_after, label='Validation Accuracy')  # 画出val_acc数据
        # plt.title('Training and Validation Accuracy')  # 设置标题
        # plt.legend()  # 画出图例
        #
        # plt.subplot(1, 2, 2)  # 画出第二列
        # plt.plot(train_loss_after, label='Training Loss')
        # plt.plot(val_loss_after, label='Validation Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.show(block=False)
        # plt.pause(1)

        model,modelname= Model(modelnum=modelnum,nb_classes=nb_classes,Chans=Chans,Samples=Samples)
        model.load_weights(checkpointer_filefinal)
        probs = model.predict(test_x)
        preds = probs.argmax(axis=-1)
        test_acc = round(100*np.mean(preds == test_y.argmax(axis=-1)),2)

        print("Extra_train Classification accuracy  : %.2f " % (test_acc))

    print("%d-fold Classification accuracy  : %.2f " % (kfoldnum, test_acc_before))

    print("%d-fold All_AccuracyTrain : "%(kfoldnum),All_AccuracyTrain)
    print("---------mean acc:%.2f  std:%.2f "%(np.mean(All_AccuracyTrain),np.std(All_AccuracyTrain)))
    print("%d-fold All_AccuracyVal : "%(kfoldnum),All_AccuracyVal)
    print("---------mean acc:%.2f  std:%.2f "%(np.mean(All_AccuracyVal),np.std(All_AccuracyVal)))
    print("%d-fold All_AccuracyTest : "%(kfoldnum),All_AccuracyTest)
    print("---------mean acc:%.2f  std:%.2f "%(np.mean(All_AccuracyTest),np.std(All_AccuracyTest)))

    info['n_epochs_kfold'].append(np.mean(All_epochs))
    info['best_model'].append(model)
    info['fold_accuracy_train'].append(np.mean(All_AccuracyTrain))
    info['fold_accuracy_val'].append(np.mean(All_AccuracyVal))
    info['test_accuracy_after'].append(test_acc)
    # from scipy.io import savemat
    # savemat(file_name, {"info": info})
    return info, modelname, test_acc_before, test_acc


if __name__ == '__main__':
    #  1: ShallowConvNet   2:DeepConvNet   3:EEGNet8-2
    #  4:EEGNet4-2         5:EEG_TCNet     6:EEG_inception
    #  7:EEG_ITNet         8:MI_EEGNet     9:my_model1
    #  10:my_model2

    #opt=1 adam    =2sgd
    train_accs = []
    test_accs = []
    #保存额外训练之前的精度
    before_l_r = []
    before_l_f = []
    before_l_t = []
    before_r_f = []
    before_r_t = []
    before_f_t = []
    accs_before = []

    #保存额外训练之后的精度
    l_r = []
    l_f = []
    l_t = []
    r_f = []
    r_t = []
    f_t = []
    accs = []

    #设置分类类别
    nb_classes = 4
    #设置训练模型
    modelnum = 8
    #设置k-fold
    kfoldnum = 5
    #设置训练批次
    epochs = 20
    #设置监控批次数  如果Es=false 这个值没有作用
    Es = True
    patience = 100
    #设置额外训练批次数  如果extra_train=false 这个值没有作用
    extra_train = True
    extra_epoch = 20
    # 无二分类
    con = 1
    if nb_classes == 2:
        con = 6
    subject_ids = [6]
    for class_id in range(con):
        for subject_id in subject_ids:
            info,modelname,test_acc_before,test_acc = Kfoldtrain(class_id,subject_id,
                                                                nb_classes=nb_classes,modelnum = modelnum,opt=1,
                                                                kfoldnum = kfoldnum, epochs = epochs, Es=Es,patience=patience,
                                                                extra_train=extra_train, extra_epoch = extra_epoch,resamplie=True)
            accs_before.append(test_acc_before)
            accs.append(test_acc)





# def toarray(data):
#     data = np.squeeze(np.array(data))
#     return data

# import matplotlib.pyplot as plt
# train_acc = toarray(info['train_acc_after'])
# train_loss = toarray(info['train_loss_after'])
# val_acc = toarray(info['val_acc_after'])
# val_loss = toarray(info['val_loss_after'])
#
# plt.subplot(1, 2, 1)  # 将图像分为1行2列，这段代码，画出第1列
# plt.plot(train_acc, label='Training Accuracy')  # 画出acc数据
# plt.plot(val_acc, label='Validation Accuracy')  # 画出val_acc数据
# plt.title('Training and Validation Accuracy')  # 设置标题
# plt.legend()  # 画出图例
#
# plt.subplot(1, 2, 2)  # 画出第二列
# plt.plot(train_loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()
