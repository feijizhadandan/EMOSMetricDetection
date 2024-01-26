from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.metric.model import MyDataset, segment
from dao.DetectionResultDAO import DetectionResultDAO
from dao.DetectionScoreResultDAO import DetectionScoreResultDAO
from dao.DetectionTaskDAO import DetectionTaskDAO
from entity.DetectionResult import DetectionResult
from entity.DetectionScoreResult import DetectionScoreResult
from entity.DetectionTask import DetectionTask
from utils import saveModelUtil
from utils.prometheusUtil import PROMETHEUS


def detectOnline(threadName: str, detectParam: dict, detectionTask: DetectionTask):

    # 加载的模型
    load_model, load_podNames, load_metricNames, load_threshold = saveModelUtil.loadModelByName(detectionTask.modelName)
    detectPodList = detectParam.get("podValue")
    # 搜集预测的数据
    metricMap = PROMETHEUS.getMetrics(load_metricNames, load_podNames, startTime=detectionTask.startTime.timestamp(), endTime=detectionTask.endTime.timestamp())
    PROMETHEUS.fillMetricsWithPreviousValue(metricMap)

    # 每个pod单独检测
    for podItemName in detectPodList:
        detectSingly(podItemName, detectionTask, metricMap)

    real_x = []
    for podName in load_podNames:
        for metricName in load_metricNames:
            real_x.append(metricMap.get(metricName).get(podName))

    real_x = np.array(real_x)
    real_x = np.column_stack(real_x).tolist()

    tmp_x = [[0] * len(load_metricNames) * len(load_podNames) for _ in range(50)]
    real_x = tmp_x + real_x

    window_size = 100
    step_size = 50
    real_x_windows = segment(real_x, window_size, step_size)
    dataset = MyDataset(real_x_windows, real_x_windows)

    device = torch.device('cuda:0')
    load_model.to(device)
    load_model.eval()
    # 只有一批
    data_loader = DataLoader(dataset, real_x_windows.shape[0], shuffle=True, drop_last=True)
    real_score = np.array([])
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_x_real, _ = tuple(t.to(device) for t in batch)
            real_rebuild, real_root = load_model(input_x_real, input_x_real[:, window_size - step_size:window_size, :])
            real_score = torch.pow(
                torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2), dim=2),
                0.5)

    # 检测是否有异常
    exists_anomalous = torch.any(real_score > load_threshold)

    # podName="-" 表示是总的检测任务
    DetectionResultDAO().insert(DetectionResult(
        taskId=detectionTask.taskId,
        podName="-",
        abnormal=exists_anomalous.item(),
        maxScore=real_score.max().item(),
        threshold=load_threshold
    ))

    # 检测结果
    print("异常score阈值:", load_threshold)
    print("检测范围内最大score:", real_score.max().item())
    print("检测范围内小score:", real_score.min().item())
    print("是否存在异常:", exists_anomalous.item())

    detectionTask.status = "stopped"
    DetectionTaskDAO().update(detectionTask)


def detectSingly(podName, detectionTask: DetectionTask, metricMap):
    # 加载的模型
    load_model, load_podNames, load_metricNames, load_threshold = saveModelUtil.loadModelByName(
        detectionTask.modelName + "-" + podName)

    real_x = []
    for metricName in load_metricNames:
        real_x.append(metricMap.get(metricName).get(podName))
    real_x = np.array(real_x)
    real_x = np.transpose(real_x).tolist()

    tmp_x = [[0] * len(load_metricNames) for _ in range(50)]
    real_x = tmp_x + real_x

    window_size = 100
    step_size = 50
    real_x_windows = segment(real_x, window_size, step_size)
    dataset = MyDataset(real_x_windows, real_x_windows)

    device = torch.device('cuda:0')
    load_model.to(device)
    load_model.eval()
    # 只有一批
    data_loader = DataLoader(dataset, real_x_windows.shape[0], shuffle=True, drop_last=True)
    real_score = np.array([])
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_x_real, _ = tuple(t.to(device) for t in batch)
            real_rebuild, real_root = load_model(input_x_real, input_x_real[:, window_size - step_size:window_size, :])
            real_score = torch.pow(
                torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2), dim=2),
                0.5)

    score_time = detectionTask.startTime
    detectionScoreResultList = []
    # 记录每个时间窗内的最大score
    for score_window in real_score:
        max_score = max(score_window).item()
        scoreResult = DetectionScoreResult(
            taskId=detectionTask.taskId,
            podName=podName,
            score=max_score,
            startTime=score_time
        )
        detectionScoreResultList.append(scoreResult)
        score_time += timedelta(seconds=50)
    DetectionScoreResultDAO().bulkInsert(detectionScoreResultList=detectionScoreResultList)

    # 检测是否有异常
    exists_anomalous = torch.any(real_score > load_threshold)

    DetectionResultDAO().insert(DetectionResult(
        taskId=detectionTask.taskId,
        podName=podName,
        abnormal=exists_anomalous.item(),
        maxScore=real_score.max().item(),
        threshold=load_threshold
    ))

    # 检测结果
    print("异常score阈值:", load_threshold)
    print("检测范围内最大score:", real_score.max().item())
    print("检测范围内小score:", real_score.min().item())
    print("是否存在异常:", exists_anomalous.item())
