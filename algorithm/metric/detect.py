import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.metric.model import MyDataset, segment
from entity.DetectionTask import DetectionTask
from utils import saveModelUtil
from utils.prometheusUtil import PROMETHEUS


def detectOnline(threadName: str, detectParam: dict, detectionTask: DetectionTask):

    # 加载的模型
    load_model, load_podNames, load_metricNames, load_threshold = saveModelUtil.loadModelByName(detectionTask.modelName)
    # 搜集预测的数据
    metricMap = PROMETHEUS.getMetrics(load_metricNames, startTime=detectionTask.startTime.timestamp(), endTime=detectionTask.endTime.timestamp())
    PROMETHEUS.fillMetricsWithPreviousValue(metricMap)
    real_x = []
    for podName in load_podNames:
        temList = []
        for metricName in load_metricNames:
            temList.append(metricMap.get(metricName).get(podName))
        temList = np.column_stack(temList)
        real_x.append(temList)
    real_x = np.array(real_x)
    real_x = np.reshape(real_x, (real_x.shape[0] * real_x.shape[1], real_x.shape[2]))

    window_size = 100
    step_size = 50
    real_x_windows = segment(real_x, window_size, step_size)
    dataset = MyDataset(real_x_windows, real_x_windows)

    device = torch.device('cuda:0')
    load_model.to(device)
    load_model.eval()
    data_loader = DataLoader(dataset, real_x_windows.shape[0], shuffle=True, drop_last=True)
    real_score = np.array([])
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_x_real, _ = tuple(t.to(device) for t in batch)
            real_rebuild, real_root = load_model(input_x_real, input_x_real[:, window_size - step_size:window_size, :])
            real_score = torch.pow(
                torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2), dim=2),
                0.5)

    # 检测结果
    print("异常score阈值:", load_threshold)
    print("检测范围内最大score:", real_score.max().item())
    print("检测范围内小score:", real_score.min().item())
    # 检测是否有异常
    exists_anomalous = torch.any(real_score > load_threshold)
    print("是否存在异常:", exists_anomalous.item())
