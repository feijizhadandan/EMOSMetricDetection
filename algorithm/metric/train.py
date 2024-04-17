import datetime
import json
import random
import threading

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
import torch
from sklearn import metrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio

from dao.ModelMetadataDAO import ModelMetadataDAO
from dao.TrainTaskDAO import TrainTaskDAO
from entity.ModelMetadata import ModelMetadata
from entity.TrainTask import TrainTask
from algorithm.metric.model import liner_ae, MyDataset, segment, metric_loss, evalution_roc
from utils.logUtil import LOG
from utils.prometheusUtil import PROMETHEUS
from utils import saveModelUtil
from utils.threadUtil import threadUtil


def onlineTrain(threadName: str, trainParameters: dict, trainTask: TrainTask):
    train_startTimestamp = trainTask.startTime.timestamp()
    train_endTimestamp = trainTask.endTime.timestamp()
    # 数据来源
    podList = trainParameters.get('podValue')

    # 搜集metric训练数据
    metricList = trainParameters.get("metricList")
    metricMap = PROMETHEUS.getMetrics(metricList, podList, startTime=train_startTimestamp, endTime=train_endTimestamp)
    PROMETHEUS.fillMetricsWithPreviousValue(metricMap)

    # 每个pod训练一个模型
    for podItemName in podList:
        trainSingly(podItemName, metricList, trainTask, metricMap)

    # # 每个pod训练一个模型
    # for podItemName in podList:
    #     singleTrainingThread = threading.Thread(
    #         target=trainSingly,
    #         args=(podItemName, metricList, trainTask, metricMap)
    #     )
    #     singleTrainingThread.start()

    # 数据处理(处理成数组)
    train_x = []
    for podName in podList:
        for metricName in metricList:
            train_x.append(metricMap.get(metricName).get(podName))

    scaler_list = []
    train_x = np.column_stack(train_x)
    train_x = np.array(train_x)

    n = (train_x.shape[1] - 2) // 4  # 计算n的值

    for x in range(n + 1):
        start_index = 4 * x + 2
        end_index = 4 * x + 4

        scaler = MinMaxScaler()
        # 对指定列进行归一化
        train_x[:, start_index:end_index] = scaler.fit_transform(train_x[:, start_index:end_index])
        train_x[:, start_index - 1] *= 1000
        train_x[:, start_index] *= 5
        train_x[:, start_index + 1] *= 5
        scaler_list.append(scaler)

    # 处理好数据后，计算每个指标的均值m和标准差sigma
    mean_list = np.mean(train_x, axis=0)
    sigma_list = np.std(train_x, axis=0)

    # smd
    num_head = 2
    # 窗口大小为100
    window_size = 100
    step_size = 50
    # dim=train_x.shape[1]
    # 样本数量必须>100
    train_x_windows = segment(train_x, window_size, step_size)
    device = torch.device('cpu')
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
        LOG.logger.info(f"total-训练轮次: {epoch}")
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
    real_threshold = mean + 2 * std

    # 训练完后保存模型
    saveModelUtil.saveModelByName(model, epoch=epochMax, podNames=podList, metricName=metricList,
                                  threshold=real_threshold, modelName=trainTask.modelName, scalerList=scaler_list,
                                  mean_list=mean_list, sigma_list=sigma_list)

    # 将模型信息保存至数据库
    ModelMetadataDAO().insert(
        ModelMetadata(
            modelName=trainTask.modelName,
            mark=1,
            createTime=datetime.datetime.now(),
            meta=json.dumps(trainParameters)
        )
    )
    trainTask.progress = 100
    trainTask.status = "stopped"
    threadUtil.trainingThreadsMap[trainTask.taskId]["status"] = "stopped"
    # 更新数据库中线程的信息
    TrainTaskDAO().update(trainTask)


def trainSingly(podName, metricList, trainTask: TrainTask, metricMap):
    #
    # SEED = 49
    # if SEED != -1:
    #     random.seed(SEED)
    #     np.random.seed(SEED)
    #     torch.manual_seed(SEED)

    # 数据处理
    train_x = []
    for metricName in metricList:
        train_x.append(metricMap.get(metricName).get(podName))
    train_x = np.array(train_x)
    train_x = np.transpose(train_x)
    # 归一化
    scaler_list = []
    scaler = MinMaxScaler()
    # 增大memory数量级
    # 提取后两列数据
    train_x_last_two = train_x[:, -2:]
    # 对后两列数据进行归一化
    train_x_last_two_normalized = scaler.fit_transform(train_x_last_two)
    train_x[:, -2:] = train_x_last_two_normalized
    scaler_list.append(scaler)

    train_x[:, 1] *= 10000
    train_x[:, 2] *= 5
    train_x[:, 3] *= 5

    # smd
    num_head = 2
    # 窗口大小为100
    window_size = 100
    step_size = 50
    # dim=train_x.shape[1]
    # 样本数量必须>100
    train_x_windows = segment(train_x, window_size, step_size)
    device = torch.device('cpu')
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
        LOG.logger.info(f"{podName} 训练轮次: {epoch}")
        for batch_idx, batch in enumerate(traindata_loader):
            input_x, _ = tuple(t.to(device) for t in batch)
            x_rebuild, x_root = model(input_x, input_x[:, window_size - step_size:window_size, :])
            loss1 = loss0(x_rebuild, input_x[:, window_size - step_size:window_size, :], parameter=0.1)
            loss1.backward()
            optimizer1.step()
            all_loss1.append(loss1.item())
            # 计算训练数据集的score以后续获得阈值(最后一轮才计算)
            if epoch == epochMax - 1:
                x_rebuild_cpu = x_rebuild.detach().cpu()
                input_x_cpu = input_x[:, window_size - step_size:window_size, :].detach().cpu()
                eurDistance = np.linalg.norm(x_rebuild_cpu - input_x_cpu, axis=2)
                train_score.append(eurDistance)

    # 计算均值和标准差
    mean = np.mean(train_score)
    std = np.std(train_score)
    # 计算异常阈值
    real_threshold = mean + 2 * std

    podList = [podName]

    # 训练完后保存模型
    saveModelUtil.saveModelByName(model, epoch=epochMax, podNames=podList, metricName=metricList,
                                  threshold=real_threshold, modelName=trainTask.modelName + "-" + podName,
                                  scalerList=scaler_list)

    # 将模型信息保存至数据库
    ModelMetadataDAO().insert(
        ModelMetadata(
            modelName=trainTask.modelName + "-" + podName,
            mark=2,
            createTime=datetime.datetime.now()
        )
    )
