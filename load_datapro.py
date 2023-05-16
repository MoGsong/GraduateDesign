"""
此文件用于加载数据
"""
from DataAugmentation import Shift_Window
import numpy as np
import os.path
# from load_data import BCI_IV_2a
# import mne
import scipy.io as sio
# from sklearn.model_selection import train_test_split
from pro import standardize_data  # ,exponential_running_standardize


def to_fenlei(fenlei):
    # 二分类任务
    #'769': left, '770': right, '771': foot, '772': tongue
    #'769': 1  ,  '770': 2  ,   '771': 3  ,  '772': 4
    if fenlei==1:
        event_id = dict({'769': 1 , '770': 2})
        id = [1, 2]
    elif fenlei==2:
        event_id = dict({'769': 1 , '771': 3})
        id = [1, 3]
    elif fenlei==3:
        event_id = dict({'769': 1 , '772': 4})
        id = [1, 4]
    elif fenlei==4:
        event_id = dict({'770': 2  ,'771': 3})
        id = [2, 3]
    elif fenlei==5:
        event_id = dict({'770': 2  ,'772': 4})
        id = [2, 4]
    elif fenlei==6:
        event_id = dict({'771': 3  ,'772': 4})
        id = [3, 4]
    else:
        print("you must to set the class")
    return event_id,id

def to_labels(raw_data,id):
    j=0
    labels = np.zeros(144)
    for i in range(len(raw_data.info["events"][:,2])):
        if raw_data.info["events"][:,2][i] == id[0] or raw_data.info["events"][:,2][i] == id[1]:
            labels[j] = raw_data.info["events"][:,2][i]
            j = j+1
    return labels


def Pre_Data(class_id, subject_id, nb_classes, standardize=1, resamplie=False):
    ###从mat里直接导入数据，然后将数据按照一定的规则进行划分，这里建议用这个

    if nb_classes == 2:
        # 二分类处理
        makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/to_CNN_class2/" + makers[class_id])
        print('------', makers[class_id] + '/A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    elif nb_classes == 4:
        # 四分类处理
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/to_CNN_class4/")
        print('------', 'A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

        # 频率
        Fs = 250
        trX = mat_data['train_data']
        # 标准化
        if standardize == 1:
            trX = standardize_data(trX, Fs)
        elif standardize == 2:
            from sklearn.preprocessing import StandardScaler
            for j in range(22):
                scaler = StandardScaler()
                scaler.fit(trX[:, j, :])
                trX[:, j, :] = scaler.transform(trX[:, j, :])

        trY = mat_data['train_labels']  # (144 2)

        teX = mat_data['test_data']
        if standardize == 1:
            teX = standardize_data(teX, Fs)
        elif standardize == 2:
            from sklearn.preprocessing import StandardScaler
            for j in range(22):
                scaler = StandardScaler()
                scaler.fit(teX[:, j, :])
                teX[:, j, :] = scaler.transform(teX[:, j, :])

        teY = mat_data['test_labels']

        # 降采样 128
        if resamplie:
            from scipy import signal
            trX = signal.resample_poly(trX, 128, 250, axis=2)
            teX = signal.resample_poly(teX, 128, 250, axis=2)

        # train_x, val_x, train_y, val_y = train_test_split(trX, trY, test_size=0.20, random_state=42, shuffle=True)
        test_x = teX
        test_y = teY

        # return (train_x,train_y),(val_x,val_y),(test_x,test_y)
        return (trX, trY), (test_x, test_y)

def SW_Pre_Data(class_id, subject_id, nb_classes, standardize=1, resamplie=False):
    ###从mat里直接导入数据，然后将数据按照一定的规则进行划分，这里建议用这个

    if nb_classes == 2:
        # 二分类处理
        makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/SlideWindow_class2/" + makers[class_id])
        print('------', makers[class_id] + '/A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    elif nb_classes == 4:
        # 四分类处理
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/SlideWindow_class4/")
        print('------', 'A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    trX = mat_data['train_data']  # (144,22,500)
    _, Chans, Samples = trX.shape
    # 频率
    Fs = 250
    # 标准化
    if standardize == 1:
        trX = standardize_data(trX, Fs)
    elif standardize == 2:
        from sklearn.preprocessing import StandardScaler
        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(trX[:, j, :])
            trX[:, j, :] = scaler.transform(trX[:, j, :])

    trY = mat_data['train_labels']  # (144 2)

    teX = mat_data['test_data']
    if standardize == 1:
        teX = standardize_data(teX, Fs)
    elif standardize == 2:
        from sklearn.preprocessing import StandardScaler
        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(teX[:, j, :])
            teX[:, j, :] = scaler.transform(teX[:, j, :])

    teY = mat_data['test_labels']

    # 降采样 128
    if resamplie:
        from scipy import signal
        trX = signal.resample_poly(trX, 128, 250, axis=2)
        teX = signal.resample_poly(teX, 128, 250, axis=2)

    # train_x, val_x, train_y, val_y = train_test_split(trX, trY, test_size=0.20, random_state=42, shuffle=True)
    test_x = teX
    test_y = teY

    # return (train_x,train_y),(val_x,val_y),(test_x,test_y)
    return (trX, trY), (test_x, test_y)

def BCI_IV_2a_mat(class_id,subject_id,nb_classes,standardize=1,resamplie=False):
    ###从mat里直接导入数据，然后将数据按照一定的规则进行划分，这里建议用这个

    if nb_classes==2:
        #二分类处理
        makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/to_CNN_class2/" + makers[class_id])
        print('------', makers[class_id] + '/A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))
    
    elif nb_classes==4:
        #四分类处理
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/to_CNN_class4/" )
        print('------','A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    #频率
    Fs = 250

    trX = mat_data['train_data']   #(144,22,500)
    _, Chans, Samples= trX.shape

    #标准化
    if standardize ==1:
        trX = standardize_data(trX, Fs)
    elif standardize==2:
        from sklearn.preprocessing import StandardScaler
        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(trX[:,j,:])
            trX[:,j,:] = scaler.transform(trX[:,j,:])


    trX = trX.reshape(-1, Chans, Samples, 1)   #(144,22,500,1)
    trY = mat_data['train_labels']  # (144 2)




    teX = mat_data['test_data']
    if standardize ==1:
        teX = standardize_data(teX, Fs)
    elif standardize == 2:
        from sklearn.preprocessing import StandardScaler
        for j in range(22):
            scaler = StandardScaler()
            scaler.fit(teX[:, j, :])
            teX[:, j, :] = scaler.transform(teX[:, j, :])
    teX = teX.reshape(-1, Chans, Samples, 1)
    teY = mat_data['test_labels']

    #降采样 128
    if resamplie:
        from scipy import signal
        trX = signal.resample_poly(trX, 128, 250,axis=2)
        teX = signal.resample_poly(teX, 128, 250, axis=2)



    # train_x, val_x, train_y, val_y = train_test_split(trX, trY, test_size=0.20, random_state=42, shuffle=True)
    test_x = teX
    test_y = teY

    # return (train_x,train_y),(val_x,val_y),(test_x,test_y)
    return (trX,trY),(test_x,test_y)


def SW_BCI_IV_2a_mat(class_id, subject_id, nb_classes, standardize=1, resamplie=False):
    ###从mat里直接导入数据，然后将数据按照一定的规则进行划分，这里建议用这个

    if nb_classes == 2:
        # 二分类处理
        makers = ['l_r', 'l_f', 'l_t', 'r_f', 'r_t', 'f_t']
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/SlideWindow_class2/" + makers[class_id])
        print('------', makers[class_id] + '/A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    elif nb_classes == 4:
        # 四分类处理
        mkpath = os.path.join(os.getcwd() + "/data/BCI4_2a/SlideWindow_class4/")   # /data/BCI4_2a/SlideWindowDA_class4/ 保存的已数据增强文件, /data/BCI4_2a/SlideWindowDA_class4/ 运行时使用python代码进行增强
        print('------', 'A0%d.mat' % (subject_id))
        mat_data = sio.loadmat(mkpath + '/A0%d.mat' % (subject_id))

    # 频率
    Fs = 250

    trX = mat_data['train_data']  # (144,22,500)
    trY = mat_data['train_labels']  # (144 2)

    # 时频+滑窗数据增强
    # print(trX.shape)
    trX, trY = Shift_Window(trX, trY)

    # print(trX.shape)

    _, Chans, Samples = trX.shape

    # 标准化
    if standardize == 1:
        trX = standardize_data(trX, Fs)
    elif standardize == 2:
        from sklearn.preprocessing import StandardScaler
        for j in range(Chans):
            scaler = StandardScaler()
            scaler.fit(trX[:, j, :])
            trX[:, j, :] = scaler.transform(trX[:, j, :])

    trX = trX.reshape(-1, Chans, Samples, 1)  # (288,22,500,1)

    teX = mat_data['test_data']
    teY = mat_data['test_labels']

    # 时频+滑窗数据增强
    teX, teY = Shift_Window(teX, teY)

    if standardize == 1:
        teX = standardize_data(teX, Fs)
    elif standardize == 2:
        from sklearn.preprocessing import StandardScaler
        for j in range(Chans):
            scaler = StandardScaler()
            scaler.fit(teX[:, j, :])
            teX[:, j, :] = scaler.transform(teX[:, j, :])
    teX = teX.reshape(-1, Chans, Samples, 1)

    if resamplie:
        from scipy import signal
        trX = signal.resample_poly(trX, 128, 250, axis=2)
        teX = signal.resample_poly(teX, 128, 250, axis=2)

    # train_x, val_x, train_y, val_y = train_test_split(trX, trY, test_size=0.20, random_state=42, shuffle=True)
    test_x = teX
    test_y = teY

    # return (train_x,train_y),(val_x,val_y),(test_x,test_y)
    return (trX, trY), (test_x, test_y)


if __name__ == '__main__':
    subject_id=1
    class_id=3
    # tmin=0.5
    # tmax=2.5
    # filter = True
    # freqmin=4
    # freqmax=40
    # event_id, id = to_fenlei(fenlei=1)
    # train_epochs,train_data,train_labels = BCI_IV_2a_todata(subject_id, classes,tmin,tmax ,filter, freqmin, freqmax)
    # print('----train_epochs----',train_epochs)
    # print('----train_epochs.info----', train_epochs.info)
    # print('----train_data----',train_data.shape)
    # print('----train_labels----',train_labels)
    # print(train_epochs.info['events'])

    (train_x,train_y),(test_x,test_y) = BCI_IV_2a_mat(class_id,subject_id,nb_classes=4,standardize=2,resamplie=True)
    print(train_x.shape)
    # print(val_x.shape)
    print(test_x.shape)
    (train_x, train_y), (test_x, test_y) = BCI_IV_2a_mat(class_id, subject_id, nb_classes=4, standardize=1,
                                                         resamplie=True)
    print(train_x.shape)
    # print(val_x.shape)
    print(test_x.shape)


