"""
此文件进行一些标准化操作
"""

import logging

import pandas as pd
import numpy as np
# import scipy
# import scipy.signal

log = logging.getLogger(__name__)



def exponential_running_standardize(
        data, factor_new=0.001, init_block_size=None, eps=1e-4
):
    """
    Perform exponential running standardization.

    Compute the exponental running mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

    Then, compute exponential running variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.


    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.

    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=other_axis, keepdims=True
        )
        init_block_standardized = (data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized

def standardize_data(data,Fs):
    splited_data = []
    for epoch in data:
        normalized_data1 = exponential_running_standardize(epoch, init_block_size=int(Fs * 4))
        splited_data.append(normalized_data1)
    splited_data = np.stack(splited_data)
    data2 = splited_data
    return data2

def spilt_data(data,fraction):
    length = len(data) * fraction
    data1 = data[:int(length),]
    data2 = data[int(length):,]
    return data1,data2