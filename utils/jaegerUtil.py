import json
from typing import List, Dict

import requests

from utils.configUtil import CONFIG
from utils.logUtil import LOG


class JaegerUtil:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def getAllServices(self) -> List[str]:
        '''
        返回jaeger探测得到的所有service
        :return: service列表
        '''
        url = f"http://{self.host}:{self.port}/api/services"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["data"]
        else:
            LOG.logger.info("Jaeger获取Service列表失败")
            return []

    def getTracesByHTTPHelper(self, startTime: int, endTime: int, limit: int) -> List[Dict]:
        '''
        通过jaeger的HTTP API获取指定时间范围内的组装好的trace，但这个接口很慢
        注：此函数会进行Trace完整性保证相关工作，如果trace是残缺的，则不会被加入到结果列表中
        :param startTime: 起始时间, 秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param limit: 结果条数
        :return: trace列表
        trace结构参见README.md中的"期望的Trace结构"部分
        '''
        traceList = []
        url = f'http://{self.host}:{self.port}/api/traces?start={startTime * 1000000}&end={endTime * 1000000}&limit={limit}&service={CONFIG["jaeger.query_service_name"]}'
        response = requests.get(url)
        print(url)
        if response.status_code == 200:
            data = response.json()
            originalTraceList = data["data"]
            for originalTrace in originalTraceList:
                spanCounter = {}  # 用于保证trace的完整性，可参见README.md中的"Trace的完整性保证"一节。
                curTrace = {}
                curTrace["traceID"] = originalTrace["traceID"]
                curTrace["startTime"] = 99999999999999999999999999999
                curTrace["spans"] = {}
                # 以下是获取span和podID之间的对应关系
                processID2PodName = {}
                for process, value in originalTrace["processes"].items():
                    processTags = value["tags"]
                    for processTag in processTags:
                        if processTag["key"] == "pod.name":
                            processID2PodName[process] = processTag["value"]
                # 以下是统计每个父span应当包含的子span个数并初始化计数器
                for span in originalTrace["spans"]:
                    if "logs" in span:
                        spanCounter[span["spanID"]] = [len(span["logs"]) / 2, 0]
                # 构建预定格式的trace
                valid = True  # trace是否合法
                for span in originalTrace["spans"]:
                    if span['startTime'] < curTrace['startTime']:
                        curTrace["startTime"] = span["startTime"]
                    if "processID" not in span:
                        valid = False
                        break
                    uri = "unknown"
                    for tag in span["tags"]:
                        if tag["key"] == "uri":
                            uri = tag["value"]
                            break
                    if uri == "unknown":
                        valid = False
                        break
                    parentSpanID = ""
                    if len(span["references"]) != 0:
                        parentSpanID = span["references"][0]["spanID"]
                        if parentSpanID not in spanCounter:
                            valid = False
                            break
                        spanCounter[parentSpanID][1] += 1
                    curTrace["spans"][span["spanID"]] = {
                        "parentSpanID": parentSpanID,
                        "startTime": span["startTime"],
                        "duration": span["duration"],
                        "podName": processID2PodName[span["processID"]],
                        "operation": uri
                    }
                for spanID in spanCounter:
                    if spanCounter[spanID][0] != spanCounter[spanID][1]:
                        valid = False
                if valid:
                    traceList.append(curTrace)
            return traceList
        else:
            LOG.logger.error("Jaeger通过HTTP获取指定时间范围内的trace失败")
            LOG.logger.error(response.content)
            return []

    def getTracesByHTTP(self, startTime: int, endTime: int, limit: int) -> List[Dict]:
        '''
        分块取数据，不然Jaeger一次取太多数据可能会炸，分块以时间窗PARTITION为准
        :param startTime: 起始时间, 秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param limit: 结果条数限制
        :return: trace列表
        '''
        traceList = []
        PARTITION = 300  # 5分钟
        blocknum = (endTime - startTime) // PARTITION
        LOG.logger.info(f"开始分段获取trace，每段{PARTITION}s，需要分${blocknum}块")
        for i in range(blocknum):
            traceList.extend(
                self.getTracesByHTTPHelper(startTime + i * PARTITION, startTime + (i + 1) * PARTITION, limit))
        if endTime - startTime - blocknum * PARTITION > 0:
            traceList.extend(self.getTracesByHTTPHelper(startTime + blocknum * PARTITION, endTime, limit))
        return traceList


    def loadJaegerTraceJson(self, path: str) -> Dict:
        '''
        加载一个jaeger格式的单条的trace json文件, 方便调试用。可以直接将从jaeger ui上得到的数据放到json文件中然后读取解析
        注：此函数不包含Trace完整性保证的验证工作
        :param path: json文件路径
        :return: 符合“期望的Trace结构”的trace
        '''
        with open(path, "r") as f:
            jsondata = json.load(f)
            originalTrace = jsondata['data'][0]
            curTrace = {}
            curTrace["traceID"] = originalTrace["traceID"]
            curTrace["startTime"] = -1
            curTrace["spans"] = {}
            # 以下是获取span和podID之间的对应关系
            processID2PodName = {}
            for process, value in originalTrace["processes"].items():
                processTags = value["tags"]
                for processTag in processTags:
                    if processTag["key"] == "pod.name":
                        processID2PodName[process] = processTag["value"]
            # 构建预定格式的trace
            valid = True  # trace是否合法
            for span in originalTrace["spans"]:
                curTrace["startTime"] = span["startTime"]
                if "processID" not in span:
                    valid = False
                    break
                uri = "unknown"
                for tag in span["tags"]:
                    if tag["key"] == "uri":
                        uri = tag["value"]
                        break
                if uri == "unknown":
                    valid = False
                    break
                curTrace["spans"][span["spanID"]] = {
                    "parentSpanID": span["references"][0]["spanID"] if len(span["references"]) != 0 else "",
                    "startTime": span["startTime"],
                    "duration": span["duration"],
                    "podName": processID2PodName[span["processID"]],
                    "operation": uri
                }
            if valid:
                return curTrace
            else:
                return {}

    def validateJaegerTrace(self, path: str) -> bool:
        '''
        加载一个jaeger格式的单条的trace json文件并利用logs字段进行Trace完整性校验
        :param path: json文件路径
        :return: 该trace是否完整
        '''
        valid = True
        with open(path, "r") as f:
            jsondata = json.load(f)
            originalTrace = jsondata['data'][0]
            spanCounter = {}  # 用于保证trace的完整性，可参见README.md中的"Trace的完整性保证"一节。
            # 以下是统计每个父span应当包含的子span个数并初始化计数器
            for span in originalTrace["spans"]:
                if "logs" in span:
                    spanCounter[span["spanID"]] = [len(span["logs"]) / 2, 0]
            for span in originalTrace["spans"]:
                if "processID" not in span:
                    valid = False
                    break
                uri = "unknown"
                for tag in span["tags"]:
                    if tag["key"] == "uri":
                        uri = tag["value"]
                        break
                if uri == "unknown":
                    valid = False
                    break
                if len(span["references"]) != 0:
                    parentSpanID = span["references"][0]["spanID"]
                    spanCounter[parentSpanID][1] += 1
            for spanID in spanCounter:
                if spanCounter[spanID][0] != spanCounter[spanID][1]:
                    valid = False
                    break
        return valid

    def visualizeTrace(self, trace: Dict):
        '''
        TODO: 将传入的一条符合"期望的Trace结构"的trace可视化为一张调用图
        :param trace:
        :return:
        '''
        pass

    def getTraceByTraceId(self, traceId: str):
        '''
        通过traceId获取trace，如果trace不完整测返回None
        :param traceId:
        :return: 符合"期望的Trace结构"的trace
        '''
        url = f"http://{self.host}:{self.port}/api/traces/{traceId}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data is None or data["data"] is None or len(data["data"]) == 0:
                return {}
            originalTrace = data["data"][0]
            spanCounter = {}  # 用于保证trace的完整性，可参见README.md中的"Trace的完整性保证"一节。
            curTrace = {}
            curTrace["traceID"] = originalTrace["traceID"]
            curTrace["startTime"] = -1
            curTrace["spans"] = {}
            # 以下是获取span和podID之间的对应关系
            processID2PodName = {}
            for process, value in originalTrace["processes"].items():
                processTags = value["tags"]
                for processTag in processTags:
                    if processTag["key"] == "pod.name":
                        processID2PodName[process] = processTag["value"]
            # 以下是统计每个父span应当包含的子span个数并初始化计数器
            for span in originalTrace["spans"]:
                if "logs" in span:
                    spanCounter[span["spanID"]] = [len(span["logs"]) / 2, 0]
            # 构建预定格式的trace
            valid = True  # trace是否合法
            for span in originalTrace["spans"]:
                curTrace["startTime"] = span["startTime"]
                if "processID" not in span:
                    valid = False
                    break
                uri = "unknown"
                for tag in span["tags"]:
                    if tag["key"] == "uri":
                        uri = tag["value"]
                        break
                if uri == "unknown":
                    valid = False
                    break
                parentSpanID = ""
                if len(span["references"]) != 0:
                    parentSpanID = span["references"][0]["spanID"]
                    if parentSpanID not in spanCounter:
                        valid = False
                        break
                    spanCounter[parentSpanID][1] += 1
                curTrace["spans"][span["spanID"]] = {
                    "parentSpanID": parentSpanID,
                    "startTime": span["startTime"],
                    "duration": span["duration"],
                    "podName": processID2PodName[span["processID"]],
                    "operation": uri
                }
            for spanID in spanCounter:
                if spanCounter[spanID][0] != spanCounter[spanID][1]:
                    valid = False
            return curTrace if valid else {}
        else:
            LOG.logger.info("Jaeger通过traceId获取trace失败")
            return {}


JAEGER = JaegerUtil(CONFIG["jaeger.host"], CONFIG["jaeger.port"])
