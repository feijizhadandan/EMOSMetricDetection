import os
import pickle
from typing import List, Tuple

import joblib
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from algorithm.metric.model import liner_ae


def loadModelByName(modelName: str):
    """
    载入模型以及模型相关的对象
    :param modelName: 模型名
    :return:
        model: 模型
        podNames: pod名列表
        threshold: 对应此模型的阈值
    """
    with open(f"./detectionmodels/{modelName}_podNames.pkl", 'rb') as f:
        podNames = pickle.load(f)
    with open(f"./detectionmodels/{modelName}_metricNames.pkl", 'rb') as f:
        metricNames = pickle.load(f)
    with open(f"./detectionmodels/{modelName}_mean.pkl", 'rb') as f:
        meanList = pickle.load(f)
    with open(f"./detectionmodels/{modelName}_sigma.pkl", 'rb') as f:
        sigmaList = pickle.load(f)
    with open(f"./detectionmodels/{modelName}_scaler.pkl", 'rb') as f:
        scalerList = pickle.load(f)
    threshold = float(np.load(f"./detectionmodels/{modelName}_threshold.npy"))
    state_dict = torch.load(f"./detectionmodels/{modelName}.pth")
    model = liner_ae(state_dict["num_blocks_encoder"], state_dict["num_blocks_decoder"], state_dict["sample_dim"],
                     state_dict["hidden_dim"], state_dict["window_size"],
                     state_dict["step_size"], state_dict["head"])
    model.load_state_dict(state_dict["model"])
    return model, podNames, metricNames, threshold, scalerList, meanList, sigmaList


def saveModelByName(
        model: liner_ae,
        epoch: int,
        podNames: List[str],
        metricName: List[str],
        threshold: float,
        modelName: str,
        scalerList: List[MinMaxScaler],
        mean_list: List[float] = None,
        sigma_list: List[float] = None):
    """
    根据名字保存模型以及模型相关的对象
    :param metricName: 指标类型
    :param scalerList: 归一化模型數組
    :param model: 模型
    :param epoch: epoch
    :param podNames: pod名列表
    :param threshold: 对应次模型的异常阈值
    :param modelName: 模型名
    :param mean_list: 每种指标的平均值
    :param sigma_list: 每种指标的标准差
    :return:
    """
    state = {
        "model": model.state_dict(),
        "num_blocks_encoder": model.num_blocks_encoder,
        "num_blocks_decoder": model.num_blocks_decoder,
        "sample_dim": model.sample_dim,
        "hidden_dim": model.hidden_dim,
        "window_size": model.window_size,
        "step_size": model.step_size,
        "head": model.head,
        "epoch": epoch
    }
    torch.save(state, f'./detectionmodels/{modelName}.pth')
    with open(f"./detectionmodels/{modelName}_podNames.pkl", "wb") as f:
        pickle.dump(podNames, f)
    with open(f"./detectionmodels/{modelName}_metricNames.pkl", "wb") as f:
        pickle.dump(metricName, f)
    with open(f"./detectionmodels/{modelName}_mean.pkl", "wb") as f:
        pickle.dump(mean_list, f)
    with open(f"./detectionmodels/{modelName}_sigma.pkl", "wb") as f:
        pickle.dump(sigma_list, f)
    with open(f"./detectionmodels/{modelName}_scaler.pkl", "wb") as f:
        pickle.dump(scalerList, f)
    np.save(f"./detectionmodels/{modelName}_threshold.npy", np.array(threshold))
