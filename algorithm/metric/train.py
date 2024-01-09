import numpy as np
from torch.optim import Adam
import torch
from sklearn import metrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from entity.TrainTask import TrainTask
from algorithm.metric.model import liner_ae, MyDataset, segment, metric_loss, evalution_roc
from utils.prometheusUtil import PROMETHEUS


def onlineTrain(threadName: str, trainingParameters: dict, trainTask: TrainTask):

    train_startTimestamp = 1704513266
    train_endTimestamp = 1704523468

    # 搜集metric训练数据
    metricList = ['cpu_usage_30s', 'memory_rate_to_machine_total', 'memory_rate_to_machine_total']
    metricMap = PROMETHEUS.getMetrics(metricList, startTime=train_startTimestamp, endTime=train_endTimestamp)
    PROMETHEUS.fillMetricsWithPreviousValue(metricMap)

    # 数据处理(处理成数组)
    train_x = []
    for podName in PROMETHEUS.podList:
        temList = []
        for metricName in metricList:
            temList.append(metricMap.get(metricName).get(podName))
        temList = np.column_stack(temList)
        train_x.append(temList)

    train_x = np.array(train_x)
    train_x = np.reshape(train_x, (train_x.shape[0] * train_x.shape[1], train_x.shape[2]))

    # smd
    num_head = 2

    # 窗口大小为100
    window_size = 100
    step_size = 50
    # dim=train_x.shape[1]
    # 样本数量必须>100
    train_x_windows = segment(train_x, window_size, step_size)

    device = torch.device('cuda:0')
    batch_size = 128
    num_blocks_encoder = 15
    num_blocks_decoder = 15
    model = liner_ae(num_blocks_encoder, num_blocks_decoder, sample_dim=train_x.shape[1], hidden_dim=2048,
                     window_size=window_size, step_size=step_size, head=num_head)
    model.to(device)

    # training dataset
    dataset = MyDataset(train_x_windows, train_x_windows)
    # 时间窗的数量必须>batch_size，否则不能形成完整的一批
    traindata_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    # training
    epochMax = 100
    optimizer1 = Adam(model.parameters(), lr=0.001)
    loss0 = metric_loss()
    all_loss1 = []
    model.train()
    train_score = []
    for epoch in range(epochMax):
        for batch_idx, batch in enumerate(traindata_loader):
            input_x, _ = tuple(t.to(device) for t in batch)
            x_rebuild, x_root = model(input_x, input_x[:, window_size - step_size:window_size, :])
            # 计算训练数据集的score以后续获得阈值(最后一轮才计算)
            if epoch == epochMax - 1:
                x_rebuild_cpu = x_rebuild.detach().cpu()
                input_x_cpu = input_x[:, window_size - step_size:window_size, :].detach().cpu()
                eurDistance = np.linalg.norm(x_rebuild_cpu - input_x_cpu, axis=2)
                train_score.append(eurDistance)
            loss1 = loss0(x_rebuild, input_x[:, window_size - step_size:window_size, :], parameter=0.1)
            loss1.backward()
            optimizer1.step()
            all_loss1.append(loss1.item())

    # 计算均值和标准差
    mean = np.mean(train_score)
    std = np.std(train_score)
    # 计算异常阈值
    real_threshold = mean + 3 * std



    # 预测数据
    model.eval()
    test_startTimestamp = 1704523220
    test_endTimestamp = 1704523460
    # 搜集metric训练数据
    metricList = ['cpu_usage_30s', 'memory_rate_to_machine_total', 'memory_rate_to_machine_total']
    metricMap = PROMETHEUS.getMetrics(metricList, startTime=test_startTimestamp, endTime=test_endTimestamp)
    PROMETHEUS.fillMetricsWithPreviousValue(metricMap)
    test_x = []
    for podName in PROMETHEUS.podList:
        temList = []
        for metricName in metricList:
            temList.append(metricMap.get(metricName).get(podName))
        temList = np.column_stack(temList)
        test_x.append(temList)
    test_x = np.array(test_x)
    test_x = np.reshape(test_x, (test_x.shape[0] * test_x.shape[1], test_x.shape[2]))

    test_x_windows = segment(test_x, window_size, step_size)
    test_score = []
    dataset = MyDataset(test_x_windows, test_x_windows)
    test_data_loader = DataLoader(dataset, test_x_windows.shape[0], shuffle=True, drop_last=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_data_loader):
            input_x_real, _ = tuple(t.to(device) for t in batch)
            real_rebuild, real_root = model(input_x_real, input_x_real[:, window_size - step_size:window_size, :])
            test_score = torch.pow(
                torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2), dim=2),
                0.5)

    # 异常阈值
    print("异常score阈值:", real_threshold)
    # 检测结果
    print("检测范围内最大score:", test_score.max().item())
    print("检测范围内小score:", test_score.min().item())

    # 检测是否有异常
    exists_anomalous = torch.any(test_score > real_threshold)

    print("是否存在异常:", exists_anomalous.item())
