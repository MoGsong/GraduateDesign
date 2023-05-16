import numpy as np

def sliding_window(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    return windows

def random_flip(window):
    if np.random.rand() > 0.5:
        return np.flip(window, axis=-1)
    else:
        return window

def crop_sliding_windows(data, window_size, step_size, num_crops):
    cropped_windows = []
    for i in range(num_crops):
        start = np.random.randint(0, data.shape[-1] - window_size + 1)
        end = start + window_size
        cropped_windows.append(data[..., start:end])
    return cropped_windows

def Shift_Window(data, labels, window_size=500, step_size=250, num_crops=5):
    augmented_data = []
    augmented_labels = []
    for i in range(data.shape[0]):
        windows = sliding_window(data[i], window_size, step_size)
        flipped_windows = []
        cropped_windows = []
        for window in windows:
            flipped_windows.append(random_flip(window))
        for j in range(num_crops):
            cropped_windows += crop_sliding_windows(data[i], window_size, step_size, 1)
        augmented_data += flipped_windows + cropped_windows
        augmented_labels += [labels[i]] * (len(flipped_windows) + len(cropped_windows))
    return np.array(augmented_data), np.array(augmented_labels)



def add_gaussian_noise(data, labels, noise_std=0.1):
    # data为三维数组（样本数 x 通道数 x 时间点数）
    # labels为一维数组，记录每个样本的标签
    # noise_std表示噪声的标准差
    noisy_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        noise = np.random.normal(scale=noise_std, size=data[i].shape)
        noisy_data[i] = data[i] + noise
    augmented_data = np.concatenate((data, noisy_data), axis=0)
    augmented_labels = np.concatenate((labels, labels), axis=0)
    return augmented_data, augmented_labels
