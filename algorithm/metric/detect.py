import copy
import threading
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.metric.model import MyDataset, segment
from dao.RootCauseResultDAO import RootCauseResultDAO
from dao.DetectionResultDAO import DetectionResultDAO
from dao.DetectionScoreRecordDAO import DetectionScoreRecordDAO
from dao.DetectionTaskDAO import DetectionTaskDAO
from entity.DetectionResult import DetectionResult
from entity.DetectionScoreRecord import DetectionScoreRecord
from entity.DetectionTask import DetectionTask
from entity.RootCauseResult import RootCauseResult
from utils import saveModelUtil
from utils.configUtil import CONFIG
from utils.logUtil import LOG
from utils.prometheusUtil import PROMETHEUS
from utils.threadUtil import threadUtil


def realTimeDataDetect(detectParam, detectionTask):
    # 加载的模型
    load_model, load_podNames, load_metricNames, load_threshold, scalerList, mean_list, sigma_list = saveModelUtil.loadModelByName(
        detectionTask.modelName)
    # 每次启动检测的时间窗单位(s)
    detectWindowSize = 50
    # 线程睡眠时间(s)
    sleepTime = detectWindowSize
    # 如果发现异常，再进行检测的时间窗个数
    abnormalWindowCntMax = CONFIG["detectionParameters"]["abnormalWindowCntMax"]
    # 当前异常的时间个数
    abnormalCnt = 0
    # 整体是否存在异常
    exists_anomalous = False
    # 上一个时间窗是否正常
    last_window_normal = True
    # 记录总体("-")的最大分数
    total_max_score = -1

    # 记录pod的整体信息
    # pod是否存在过异常
    pod_anomalous_record = [False] * len(load_podNames)
    # pod记录的分数
    pod_score_record = [-1] * len(load_podNames)
    # pod的阈值
    pod_threshold_record = [-1] * len(load_podNames)

    startTime = detectionTask.startTime
    # 时间窗开始时间
    curStartTime = startTime
    # 时间窗结束时间
    curEndTime = startTime
    # 记录每段异常开始时间
    abnormalStartTime = startTime
    # 记录每段异常结束时间
    abnormalEndTime = startTime
    rootCaseData = []
    # 是否为第一个时间窗（用于初始化数据库）
    isFirstRound = True

    while True:
        if threadUtil.detectionThreadsMap[detectionTask.taskId]["status"] == "stopped":
            break
        # 记录本时间窗是否正常
        isNowWindowNormal = True
        LOG.logger.debug(f"等待{sleepTime}s后进行数据收集")
        time.sleep(sleepTime)
        curEndTime = curStartTime + timedelta(seconds=detectWindowSize)
        LOG.logger.info(f"开始收集数据：{curStartTime}~{curEndTime}")
        # 搜集数据
        metricMap = PROMETHEUS.getMetrics(load_metricNames, load_podNames,
                                          startTime=curStartTime.timestamp(),
                                          endTime=curEndTime.timestamp())
        PROMETHEUS.fillMetricsWithPreviousValue(metricMap)
        pod_index = 0
        # 每个pod单独检测
        for podItemName in load_podNames:
            single_anomalous, single_score, single_threshold = detectSingly(podItemName, detectionTask, metricMap, True,
                                                                            curStartTime)
            pod_score_record[pod_index] = max(pod_score_record[pod_index], single_score)
            pod_threshold_record[pod_index] = single_threshold
            if single_anomalous:
                exists_anomalous = True
                pod_anomalous_record[pod_index] = True
                isNowWindowNormal = False
            pod_index += 1

        real_x = []
        for podName in load_podNames:
            for metricName in load_metricNames:
                real_x.append(metricMap.get(metricName).get(podName))

        real_x = np.array(real_x)
        real_x = np.column_stack(real_x).tolist()
        real_x = np.array(real_x)

        n = (real_x.shape[1] - 2) // 4  # 计算n的值

        for x in range(n + 1):
            start_index = 4 * x + 2
            end_index = 4 * x + 4

            # 对指定列进行归一化
            real_x[:, start_index:end_index] = scalerList[x].transform(real_x[:, start_index:end_index])
            real_x[:, start_index - 1] *= 1000
            real_x[:, start_index] *= 5
            real_x[:, start_index + 1] *= 5

        # 暂存处理好后的数据（不带0的）
        origin_data = real_x
        tmp_x = np.zeros((50, real_x.shape[1]))
        real_x = np.vstack((tmp_x, real_x))

        window_size = 100
        step_size = 50
        real_x_windows = segment(real_x, window_size, step_size)
        dataset = MyDataset(real_x_windows, real_x_windows)

        device = torch.device('cpu')
        load_model.to(device)
        load_model.eval()
        # 只有一批
        data_loader = DataLoader(dataset, real_x_windows.shape[0], drop_last=True)
        real_score = np.array([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_x_real, _ = tuple(t.to(device) for t in batch)
                real_rebuild, real_root = load_model(input_x_real,
                                                     input_x_real[:, window_size - step_size:window_size, :])
                real_score = torch.pow(
                    torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2),
                              dim=2),
                    0.5)

        score_time = detectionTask.startTime
        detectionScoreResultList = []
        score_in_window_list = []
        # 记录每个时间窗内的最大score
        for score_window in real_score:
            abnormal_cnt_arr = score_window[score_window > load_threshold]
            # 记录每个时间窗内异常/正常时间戳的个数
            normal_cnt = 50 - len(abnormal_cnt_arr)
            abnormal_cnt = len(abnormal_cnt_arr)
            score_in_window = torch.mean(score_window).item()
            score_in_window_list.append(score_in_window)
            if score_in_window > load_threshold:
                exists_anomalous = True
                isNowWindowNormal = False
            scoreResult = DetectionScoreRecord(
                taskId=detectionTask.taskId,
                podName="-",
                score=score_in_window,
                startTime=score_time,
                normalCnt=normal_cnt,
                abnormalCnt=abnormal_cnt
            )
            detectionScoreResultList.append(scoreResult)
            score_time += timedelta(seconds=50)
        total_max_score = max(total_max_score, max(score_in_window_list))
        DetectionScoreRecordDAO().bulkInsert(detectionScoreRecordList=detectionScoreResultList)

        # 如果本时间窗是异常的
        if not isNowWindowNormal:
            # 如果上个时间窗是正常的，则标记异常开始时间
            if last_window_normal:
                abnormalStartTime = curStartTime
            abnormalCnt += 1
            rootCaseData.append(origin_data)
            # 标记异常结束时间
            abnormalEndTime = curStartTime + timedelta(seconds=detectWindowSize)
            # 连续异常窗口数量到5个，则启动一轮根因定位
            if abnormalCnt == abnormalWindowCntMax:
                copyCauseData = copy.deepcopy(rootCaseData)
                onlineRootCauseThread = threading.Thread(
                    target=onlineRootCauseAnalyze,
                    args=(
                        detectionTask.taskId, abnormalStartTime, abnormalEndTime, copyCauseData, mean_list, sigma_list,
                        load_metricNames, load_podNames),
                )
                onlineRootCauseThread.start()
                # 重新计数
                abnormalCnt = 0
                rootCaseData = []
                abnormalStartTime = abnormalEndTime
            last_window_normal = False
        else:
            # 如果本时间窗正常，上个时间窗异常，则直接根因定位
            if not last_window_normal:
                rootCaseData.append(origin_data)
                abnormalEndTime = curStartTime + timedelta(seconds=detectWindowSize)
                copyCauseData = copy.deepcopy(rootCaseData)
                onlineRootCauseThread = threading.Thread(
                    target=onlineRootCauseAnalyze,
                    args=(
                        detectionTask.taskId, abnormalStartTime, abnormalEndTime, copyCauseData, mean_list, sigma_list,
                        load_metricNames, load_podNames),
                )
                onlineRootCauseThread.start()
                # 重新计数
                abnormalCnt = 0
                rootCaseData = []
            last_window_normal = True

        curStartTime += timedelta(seconds=detectWindowSize)

        if isFirstRound:
            isFirstRound = False
            for i in range(len(load_podNames)):
                DetectionResultDAO().insert(DetectionResult(
                    taskId=detectionTask.taskId,
                    podName=load_podNames[i],
                    abnormal=pod_anomalous_record[i],
                    maxScore=pod_score_record[i],
                    threshold=pod_threshold_record[i]
                ))

            DetectionResultDAO().insert(DetectionResult(
                taskId=detectionTask.taskId,
                podName="-",
                abnormal=exists_anomalous,
                maxScore=total_max_score,
                threshold=load_threshold
            ))

            detectionTask.endTime = curEndTime
            DetectionTaskDAO().update(detectionTask)
        else:
            for i in range(len(load_podNames)):
                DetectionResultDAO().updateByTaskIdAndPodName(DetectionResult(
                    taskId=detectionTask.taskId,
                    podName=load_podNames[i],
                    abnormal=pod_anomalous_record[i],
                    maxScore=pod_score_record[i],
                    threshold=pod_threshold_record[i]
                ))

            DetectionResultDAO().updateByTaskIdAndPodName(DetectionResult(
                taskId=detectionTask.taskId,
                podName="-",
                abnormal=exists_anomalous,
                maxScore=total_max_score,
                threshold=load_threshold
            ))

            detectionTask.endTime = curEndTime
            DetectionTaskDAO().update(detectionTask)


def onlineRootCauseAnalyze(taskId, abnormalStartTime, abnormalEndTime, rootCaseData, mean_list, sigma_list,
                           load_metricNames, load_podNames):
    # 处理rootCaseData数据格式
    rootCaseData = np.concatenate(rootCaseData, axis=0)
    root_score = abs(rootCaseData - mean_list) / mean_list
    median_list = np.median(root_score, axis=0)
    # 降序排列
    sorted_indices = np.argsort(median_list)[::-1]
    metric_cnt = len(load_metricNames)
    LOG.logger.info("根因定位TOP5：")
    for i in range(5):
        pod_idx = sorted_indices[i] // metric_cnt
        metric_idx = sorted_indices[i] % metric_cnt
        LOG.logger.info(f"{i + 1}: {load_podNames[pod_idx]}({load_metricNames[metric_idx]})")

    RootCauseResultDAO().insert(
        RootCauseResult(
            taskId=taskId,
            top_1=load_podNames[sorted_indices[0] // metric_cnt] + "(" + load_metricNames[
                sorted_indices[0] % metric_cnt] + ")",
            top_2=load_podNames[sorted_indices[1] // metric_cnt] + "(" + load_metricNames[
                sorted_indices[1] % metric_cnt] + ")",
            top_3=load_podNames[sorted_indices[2] // metric_cnt] + "(" + load_metricNames[
                sorted_indices[2] % metric_cnt] + ")",
            top_4=load_podNames[sorted_indices[3] // metric_cnt] + "(" + load_metricNames[
                sorted_indices[3] % metric_cnt] + ")",
            top_5=load_podNames[sorted_indices[4] // metric_cnt] + "(" + load_metricNames[
                sorted_indices[4] % metric_cnt] + ")",
            startTime=abnormalStartTime,
            endTime=abnormalEndTime
        )
    )


def detectStart(threadName: str, detectParam: dict, detectionTask: DetectionTask):
    # 是否为实时在线检测
    isDetectOnline = detectionTask.onlineTask

    if isDetectOnline:
        realTimeDataDetect(detectParam, detectionTask)
    else:
        # 加载的模型
        load_model, load_podNames, load_metricNames, load_threshold, scalerList, mean_list, sigma_list = saveModelUtil.loadModelByName(
            detectionTask.modelName)
        # 必须选择所有的pod进行检测，因为指标数量不能变化
        detectPodList = detectParam.get("podValue")
        # 搜集预测的数据
        metricMap = PROMETHEUS.getMetrics(load_metricNames, load_podNames,
                                          startTime=detectionTask.startTime.timestamp(),
                                          endTime=detectionTask.endTime.timestamp())
        PROMETHEUS.fillMetricsWithPreviousValue(metricMap)

        exists_anomalous = False
        # 每个pod单独检测
        for podItemName in load_podNames:
            single_exists_anomalous, _, _ = detectSingly(podItemName, detectionTask, metricMap, False)
            if single_exists_anomalous:
                exists_anomalous = True

        real_x = []
        for podName in load_podNames:
            for metricName in load_metricNames:
                real_x.append(metricMap.get(metricName).get(podName))

        real_x = np.array(real_x)
        real_x = np.column_stack(real_x).tolist()
        real_x = np.array(real_x)

        n = (real_x.shape[1] - 2) // 4  # 计算n的值

        for x in range(n + 1):
            start_index = 4 * x + 2
            end_index = 4 * x + 4

            # 对指定列进行归一化
            real_x[:, start_index:end_index] = scalerList[x].transform(real_x[:, start_index:end_index])
            real_x[:, start_index - 1] *= 1000
            real_x[:, start_index] *= 5
            real_x[:, start_index + 1] *= 5

        # 暂存处理好后的数据（不带0的）
        origin_data = real_x
        tmp_x = np.zeros((50, real_x.shape[1]))
        real_x = np.vstack((tmp_x, real_x))

        window_size = 100
        step_size = 50
        real_x_windows = segment(real_x, window_size, step_size)
        dataset = MyDataset(real_x_windows, real_x_windows)

        device = torch.device('cpu')
        load_model.to(device)
        load_model.eval()
        # 只有一批
        data_loader = DataLoader(dataset, real_x_windows.shape[0], drop_last=True)
        real_score = np.array([])
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_x_real, _ = tuple(t.to(device) for t in batch)
                real_rebuild, real_root = load_model(input_x_real,
                                                     input_x_real[:, window_size - step_size:window_size, :])
                real_score = torch.pow(
                    torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2),
                              dim=2),
                    0.5)

        a = real_score.cpu().numpy()
        total_exists_anomalous = False
        score_time = detectionTask.startTime
        detectionScoreResultList = []
        score_in_window_list = []
        # 记录每个时间窗内的最大score
        for score_window in real_score:
            abnormal_cnt_arr = score_window[score_window > load_threshold]
            # 记录每个时间窗内异常/正常时间戳的个数
            normal_cnt = 50 - len(abnormal_cnt_arr)
            abnormal_cnt = len(abnormal_cnt_arr)
            score_in_window = torch.mean(score_window).item()
            score_in_window_list.append(score_in_window)
            if score_in_window > load_threshold:
                exists_anomalous = True
                total_exists_anomalous = True
            scoreResult = DetectionScoreRecord(
                taskId=detectionTask.taskId,
                podName="-",
                score=score_in_window,
                startTime=score_time,
                normalCnt=normal_cnt,
                abnormalCnt=abnormal_cnt
            )
            detectionScoreResultList.append(scoreResult)
            score_time += timedelta(seconds=50)
        DetectionScoreRecordDAO().bulkInsert(detectionScoreRecordList=detectionScoreResultList)

        # 存在异常则进行根因定位
        if exists_anomalous:
            root_score = abs(origin_data - mean_list) / mean_list
            median_list = np.median(root_score, axis=0)
            # 降序排列
            sorted_indices = np.argsort(median_list)[::-1]
            metric_cnt = len(load_metricNames)
            print("根因定位TOP10：")
            for i in range(len(sorted_indices)):
                if i > 9:
                    break
                pod_idx = sorted_indices[i] // metric_cnt
                metric_idx = sorted_indices[i] % metric_cnt
                print(i + 1, ": ", load_podNames[pod_idx], "(", load_metricNames[metric_idx], ")")

            RootCauseResultDAO().insert(
                RootCauseResult(
                    taskId=detectionTask.taskId,
                    top_1=load_podNames[sorted_indices[0] // metric_cnt] + "(" + load_metricNames[
                        sorted_indices[0] % metric_cnt] + ")",
                    top_2=load_podNames[sorted_indices[1] // metric_cnt] + "(" + load_metricNames[
                        sorted_indices[1] % metric_cnt] + ")",
                    top_3=load_podNames[sorted_indices[2] // metric_cnt] + "(" + load_metricNames[
                        sorted_indices[2] % metric_cnt] + ")",
                    top_4=load_podNames[sorted_indices[3] // metric_cnt] + "(" + load_metricNames[
                        sorted_indices[3] % metric_cnt] + ")",
                    top_5=load_podNames[sorted_indices[4] // metric_cnt] + "(" + load_metricNames[
                        sorted_indices[4] % metric_cnt] + ")",
                    startTime=detectionTask.startTime,
                    endTime=detectionTask.endTime
                )
            )

        # podName="-" 表示是总的检测任务
        DetectionResultDAO().insert(DetectionResult(
            taskId=detectionTask.taskId,
            podName="-",
            abnormal=total_exists_anomalous,
            maxScore=max(score_in_window_list),
            threshold=load_threshold
        ))

        detectionTask.status = "stopped"
        DetectionTaskDAO().update(detectionTask)


def detectSingly(podName, detectionTask: DetectionTask, metricMap, isOnlineDetect, onlineTime=None):
    # 加载的模型
    load_model, load_podNames, load_metricNames, load_threshold, scaler_list, mean_list, sigma_list = saveModelUtil.loadModelByName(
        detectionTask.modelName + "-" + podName)

    real_x = []
    for metricName in load_metricNames:
        real_x.append(metricMap.get(metricName).get(podName))
    real_x = np.array(real_x)
    real_x = np.transpose(real_x).tolist()

    real_x = np.array(real_x)

    # 增大memory数量级
    # 提取后两列数据
    test_x_last_two = real_x[:, -2:]
    # 对后两列数据进行归一化
    test_x_last_two_normalized = scaler_list[0].transform(test_x_last_two)
    # 将归一化后的数据更新到原始数组中的后两列
    real_x[:, -2:] = test_x_last_two_normalized
    real_x[:, 1] *= 10000
    real_x[:, 2] *= 5
    real_x[:, 3] *= 5

    tmp_x = np.zeros((50, real_x.shape[1]))
    real_x = np.vstack((tmp_x, real_x))

    window_size = 100
    step_size = 50
    real_x_windows = segment(real_x, window_size, step_size)
    dataset = MyDataset(real_x_windows, real_x_windows)

    device = torch.device('cpu')
    load_model.to(device)
    load_model.eval()
    # 只有一批
    data_loader = DataLoader(dataset, real_x_windows.shape[0], drop_last=True)
    real_score = np.array([])
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_x_real, _ = tuple(t.to(device) for t in batch)
            real_rebuild, real_root = load_model(input_x_real, input_x_real[:, window_size - step_size:window_size, :])
            real_score = torch.pow(
                torch.sum(torch.pow(input_x_real[:, window_size - step_size:window_size, :] - real_rebuild, 2), dim=2),
                0.5)

    a = real_score.cpu().numpy()

    score_time = onlineTime if isOnlineDetect else detectionTask.startTime
    detectionScoreResultList = []
    exists_anomalous = False
    score_in_window_list = []
    # 记录每个时间窗内的最大score
    for score_window in real_score:
        abnormal_cnt_arr = score_window[score_window > load_threshold]
        # 记录每个时间窗内异常/正常时间戳的个数
        normal_cnt = 50 - len(abnormal_cnt_arr)
        abnormal_cnt = len(abnormal_cnt_arr)
        score_in_window = torch.mean(score_window).item()
        score_in_window_list.append(score_in_window)
        if score_in_window > load_threshold:
            exists_anomalous = True
        scoreResult = DetectionScoreRecord(
            taskId=detectionTask.taskId,
            podName=podName,
            score=score_in_window,
            startTime=score_time,
            normalCnt=normal_cnt,
            abnormalCnt=abnormal_cnt
        )
        detectionScoreResultList.append(scoreResult)
        score_time += timedelta(seconds=50)
    DetectionScoreRecordDAO().bulkInsert(detectionScoreRecordList=detectionScoreResultList)

    # 如果不是在线检测，则需要保存该时段总体的检测结果
    if not isOnlineDetect:
        DetectionResultDAO().insert(DetectionResult(
            taskId=detectionTask.taskId,
            podName=podName,
            abnormal=exists_anomalous,
            maxScore=max(score_in_window_list),
            threshold=load_threshold
        ))

    return exists_anomalous, max(score_in_window_list), load_threshold
