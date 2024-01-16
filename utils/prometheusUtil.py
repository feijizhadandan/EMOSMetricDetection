import json
import math
import time
from typing import List, Dict

import numpy
import requests

from utils.configUtil import CONFIG
from utils.jaegerUtil import JAEGER
from utils.logUtil import LOG


class PrometheusUtil:

    def __init__(self, host, port):
        self.v1BaseUrl = f"http://{host}:{port}/api/v1"
        self.queries = CONFIG["prometheus.queries"]

    def getAllPodNamesRelatedToTrace(self) -> List[str]:
        '''
        获取trace所涉及的所有Pod的pod name
        :return: trace所涉及的所有Pod的pod name
        '''
        serviceSet = set(JAEGER.getAllServices())  # 应当是pod name的前缀
        podSet = set()
        currentTimeStamp = int(time.time())
        getInfoUrl = f'{self.v1BaseUrl}/query_range?query={self.queries["get_info"]}&start={currentTimeStamp - 2}&end={currentTimeStamp}&step=1'
        response = requests.get(getInfoUrl)
        if response.status_code == 200:
            data = json.loads(response.text)
            results = data['data']['result']
            for result in results:
                if 'container' in result['metric'] and result['metric']['container'] in serviceSet:
                    podSet.add(result['metric']['pod'])
        else:
            LOG.logger.info(f"查询prometheus失败: {getInfoUrl}")
        return list(podSet)

    def getAllQueries(self) -> Dict:
        '''
        获取所有的PromQL查询语句
        :return: 查询语句列表
        '''
        return self.queries

    def getMetrics(self, metricList: List[str], podList: List[str], startTime: int, endTime: int, step: int = 1) -> Dict[
        str, Dict]:
        '''
        获取指定时间范围内的特定指标
        允许出现-1，当出现-1则可认为其取值为前值如需这样填充，可调用本类的fillMetricsWithPreviousValue方法
        :param podList: 目标pod名称
        :param metricList: 指标列表，取值必须为config.yaml中promethus.queries中的值
        :param startTime: 起始时间，秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param step: 步长, 默认为1s
        :return: metricMap, [指标名, [podName, 值列表]], 此外，metricMap有个key叫"startTime"，记录指标的起始时间，其类型为秒级时间戳
        '''
        # 不能一次取出过多的点位，需要分片查询
        segment = math.ceil((endTime - startTime) / 11001)  # 需要分segment段
        segSize = int((endTime - startTime + 1) // segment)  # 每段大小
        # LOG.logger.info(f"prometheus: 需要访问{segment}段，每段大小为{segSize}")
        # podNameList = self.getAllPodNamesRelatedToTrace()
        podNameList = podList
        metricMap = {
            metric: {
                podName: [-1] * int((endTime - startTime + 1) / step)
                for podName in podNameList
            }
            for metric in metricList
        }  # 默认-1，后续用请求得到的点位填充，所以如果发现有点位是-1，那么就是点位缺失了
        for i in range(0, segment - 1):  # 前面segment-1段
            for metric in metricList:
                for podName in podNameList:
                    containerName = "-".join(podName.split("-")[:-2]) + '-service'
                    currentQuery = self.queries[metric].replace("${POD_PLACEHOLDER}", podName)
                    currentQuery = currentQuery.replace("${CONTAINER_PLACEHOLDER}", containerName)
                    completeUrl = f"{self.v1BaseUrl}/query_range?query={currentQuery}&start={startTime + i * segSize}&end={startTime + (i + 1) * segSize}&step={step}"
                    response = requests.get(completeUrl)
                    if response.status_code == 200:
                        data = json.loads(response.text)
                        results = data["data"]["result"]
                        for result in results:  # 如果中间服务曾经断开重启则会出现多个result
                            values = result['values']
                            for value in values:
                                index = int((value[0] - startTime) / step)  # 计算每个点位应该在第几个位置
                                metricMap[metric][podName][index] = float(value[1])
                    else:
                        LOG.logger.info(f"查询prometheus失败: {completeUrl}")
        # 第segment段
        for metric in metricList:
            for podName in podNameList:
                containerName = "-".join(podName.split("-")[:-2]) + '-service'
                currentQuery = self.queries[metric].replace("${POD_PLACEHOLDER}", podName)
                currentQuery = currentQuery.replace("${CONTAINER_PLACEHOLDER}", containerName)
                completeUrl = f"{self.v1BaseUrl}/query_range?query={currentQuery}&start={startTime + (segment - 1) * segSize}&end={endTime}&step={step}"
                response = requests.get(completeUrl)
                if response.status_code == 200:
                    data = json.loads(response.text)
                    results = data["data"]["result"]
                    for result in results:  # 如果中间服务曾经断开重启则会出现多个result
                        values = result['values']
                        for value in values:
                            index = int((value[0] - startTime) / step)  # 计算每个点位应该在第几个位置
                            metricMap[metric][podName][index] = float(value[1])
                else:
                    LOG.logger.info(f"查询prometheus失败: {completeUrl}")
        metricMap["startTime"] = int(startTime)
        return metricMap

    def fillMetricsWithPreviousValue(self, metricMap: Dict[str, Dict]):
        '''
        将metricMap[metric][podName]列表中-1的值用前一个值补充
        特例1：如果开头是-1但并非全是-1，那么开头这段-1用之后第一个非-1的值填充
        特例2：如果整个列表全是-1，处理不了，只能返回原列表
        :param metricMap: 需要进行填充操作的指标字典
        :return: void，此函数作用于入参
        '''
        for metric in metricMap:
            if metric == "startTime":
                continue
            for podName in metricMap[metric]:
                previousValue = None
                length = len(metricMap[metric][podName])
                for i in range(length):
                    if metricMap[metric][podName][i] == -1:
                        if previousValue is not None:
                            metricMap[metric][podName][i] = previousValue
                    else:
                        previousValue = metricMap[metric][podName][i]
                # 以下代码处理特例1
                if metricMap[metric][podName][0] == -1:
                    firstValidValueIndex = None
                    for i in range(length):
                        if metricMap[metric][podName][i] != -1:
                            firstValidValueIndex = i
                            break
                    if firstValidValueIndex is not None:
                        metricMap[metric][podName][:firstValidValueIndex] = [metricMap[metric][podName][
                                                                                 firstValidValueIndex]] * firstValidValueIndex


PROMETHEUS = PrometheusUtil(CONFIG["prometheus.host"], CONFIG["prometheus.port"])
