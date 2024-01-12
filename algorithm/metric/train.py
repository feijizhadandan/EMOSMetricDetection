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
from utils import saveModelUtil
from utils.threadUtil import threadUtil


def onlineTrain(threadName: str, trainParameters: dict, trainTask: TrainTask):

    train_startTimestamp = trainTask.startTime.timestamp()
    train_endTimestamp = trainTask.endTime.timestamp()

    # 搜集metric训练数据
    metricList = trainParameters.get("metricList")
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

    # 训练完后保存模型
    saveModelUtil.saveModelByName(model, epoch=epochMax, podNames=PROMETHEUS.podList, metricName=metricList, threshold=real_threshold, modelName=trainTask.modelName)

    # TODO 将模型信息保存至数据库
    trainTask.progress = 100
    trainTask.status = "stopped"
    threadUtil.trainingThreadsMap[trainTask.taskId]["status"] = "stopped"
    # TODO 更新数据库中线程的信息


