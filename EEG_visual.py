import mne
from load_datapro import Pre_Data, SW_Pre_Data, BCI_IV_2a_mat  # , SW_BCI_IV_2a_mat
# import os
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

"""可视化数据"""
# mne直接导入读取raw
# path = r"E:\EEG\Design\Data\BCICIV_2a_gdf/"
# raw = mne.io.read_raw_gdf(path + "A02T"+".gdf", preload=True)

# 自定义raw
def eeg_plot(class_id, subject_id, nb_classes):

    # 预处理 获得数据和标签
    (train_data, train_y),(test_data, test_y) = Pre_Data(class_id, subject_id, nb_classes=nb_classes, standardize=2, resamplie=True)

    #创建info
    channel = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    info = mne.create_info(
        ch_names=channel,
        ch_types='eeg',
        sfreq=250
    )
    # 加载HMD文件并与Info对象关联
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    # 创建raw
    # print(train_data.shape)
    train_data_2d = np.transpose(train_data, (1, 0, 2)).reshape((22, -1))
    raw = mne.io.RawArray(train_data_2d, info)

    # 添加事件标记
    event_data = np.column_stack((range(len(train_y)), np.zeros(len(train_y)), train_y))
    events = np.zeros((event_data.shape[0], 3), dtype=int)
    events[:, :2] = event_data[:, :2]
    events[:, 2] = event_data[:, 2] - 1
    # print(events)
    event_id = {'left': 0, 'right': 1,'foot': 2, 'tongue': 3}
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4, baseline=None, preload=True)

    # 画地形图
    evoked = epochs.average()
    evoked.plot_topomap()
    # 画头皮电位图
    epochs.plot_sensors(show_names=True)

    # 画功率谱密度图
    epochs.plot_psd(fmin=2, fmax=40)

    # 信号图
    scalings = {'eeg': 2, 'grad': 2}
    raw.plot(scalings=scalings,
            title='train_data',
            show=True, block=False)

    # 计算时频图
    freqs = np.arange(2, 100, 2)
    n_cycles = freqs / 2.0
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=2, n_jobs=1)
    # 画时频图
    power.plot(picks='C3')     # picks='C3'

if __name__ == '__main__':
    # 参数
    nb_classes = 4
    # 二分类和四分类源数据
    con = 1
    if nb_classes == 2:
        con = 6
    # subject_id = [2, 5, 6]
    for class_id in range(con):
        for subject_id in range(1,2):
            eeg_plot(class_id, subject_id, nb_classes)
    plt.show()