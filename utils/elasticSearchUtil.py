from math import floor
from typing import List, Dict, Tuple
from elasticsearch import Elasticsearch
from utils.configUtil import CONFIG
from datetime import datetime, timedelta

from utils.logUtil import LOG

'''
用es查询只能够查询到span，而不能够查询到trace，因此需要自己实现trace的查询或者使用jaeger的api
'''
class ElasticSearchUtil:
    def __init__(self, host: str, port: str):
        self.port = port
        self.host = host
        self.client = Elasticsearch(f"http://{host}:{port}")


    def getDateRange(self, startTime: int, endTime: int) -> Dict[str, Tuple]:
        '''
        获取给定时间戳跨度内所涉及的"YYYY-MM-DD"日期以及对应的时间戳
        :param startTime: 起始时间，秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :return: <"YYYY-MM-DD", (当天的涉及起始时间戳，当天涉及的结束时间戳)
        '''
        startDate = datetime.fromtimestamp(startTime).date()
        endDate = datetime.fromtimestamp(endTime).date()
        date_range = {}
        currentDate = startDate
        currentTime = startTime
        while currentDate < endDate:
            date_range[currentDate.strftime('%Y-%m-%d')] = (currentTime, int(datetime.combine(currentDate, datetime.max.time()).timestamp()))
            currentDate += timedelta(days=1)
            currentTime = int(datetime.combine(currentDate, datetime.min.time()).timestamp())
        if currentDate == endDate:
            date_range[currentDate.strftime('%Y-%m-%d')] = (currentTime, endTime)
        return date_range


    def getSpans(self, startTime: int, endTime: int, limit: int) -> List:
        '''
        获取指定时间段内es所收集到的spans (还没测试过一次性查询过长时间跨度的数据，后果未知)
        :param startTime: 起始时间，秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param limit: 查询过程中限制的spans数量上限, -1为不限制
        :return: List里面都是spans
        '''
        # 由于jaeger对接es的时候是按照日期分出不同的index的，需要分析时间跨度涉及多少天
        dateList = self.getDateRange(startTime, endTime)
        LOG.logger.info("获取指定区间的spans")
        LOG.logger.info(f"getDateRange(startTime: {startTime}, endTime: {endTime}), dateList: {dateList}")
        queryJSON={
            "query": {
                "bool": {
                    "must":{"match_all": {}},
                    "filter":{
                        "range":{
                            "startTime":{
                                "gte": startTime*1000000, # es查询需要μs级别时间戳
                                "lte": endTime*1000000 # es查询需要μs级别时间戳
                            }
                        }
                    }
                }
            },
            "_source": True
            # "_source": [
            #     "traceID",
            #     "spanID",
            #     "operationName",
            #     "references",
            #     "startTime",
            #     "duration",
            #     "process.serviceName"
            # ]
        }
        totalSpans = []
        for date in dateList:
            if limit > 0 and len(totalSpans) > limit:
                break
            query = self.client.search(
                index='jaeger-span-{}'.format(date),
                body=queryJSON,
                scroll='1m', # 超时(es帮我们保存searchcontext的时间)，只要足够处理前一批结果就行了。每个 scroll请求都会设置一个新的过期时间。
                size=10000 # 单次返回文档数上限(rps120左右的时候1s大概就已经超过3000个span)
            )
            totalNumber = query['hits']['total']['value']
            LOG.logger.info(f"本索引共有{totalNumber}个span符合要求, 需要游标查询{totalNumber//10000+1}次")
            result = query['hits']['hits'] # es查询出的结果第一页
            totalSpans.extend(result)
            scrollID = query['_scroll_id'] # 游标用于输出es查询出的所有结果

            while len(result) > 0 and (limit == -1 or len(totalSpans) <= limit):
                LOG.logger.info(f"已收集:{len(totalSpans)}个span")
                queryScroll = self.client.scroll(scroll_id=scrollID,scroll='1m')
                scrollID = queryScroll['_scroll_id']
                result = queryScroll['hits']['hits']
                totalSpans.extend(result)
            self.client.clear_scroll(scroll_id=scrollID)
        LOG.logger.info(f"一共查询到: {len(totalSpans)}个span")
        return totalSpans

    def getPodLatenciesBetween(self, startTime: int, endTime: int, limit: int) -> Dict[str, List[int]]:
        '''
        获取指定时间段内各个pod调用的延时
        :param startTime: 起始时间，秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param limit: 查询过程中限制的spans数量上限, -1为不限制
        :return: Dict[调用关系, 延时列表]，其中调用关系为"clientPodName@serverPodName", 延时是微秒级别的
        '''
        totalSpans = self.getSpans(startTime, endTime, limit)
        spanID2Index = {} # spanID到totalSpans的index的映射
        subCallUUID2Index = {} # subCallUUID到totalSpans的index的映射
        
        for index in range(len(totalSpans)):
            curSpan = totalSpans[index]['_source']
            spanID2Index[curSpan['spanID']] =index
            if 'tags' in curSpan:
                for tag in curSpan['tags']:
                    if tag['key'] == 'subCallUUID':
                        subCallUUID2Index[tag['value']] = index
                        break
        LOG.logger.info("开始计算各个pod调用的延时")
        latencies = {}
        for index in range(len(totalSpans)):
            curSpan = totalSpans[index]['_source']
            if 'logs' in curSpan:
                subCallDict = {} # (server span的subCallUUID, client span记录的起始时间, client span记录的结束时间)
                for log in curSpan['logs']:
                    if 'fields' in log:
                        isStartLog = 0 # 0表示无效，1表示是start log，2表示是end log
                        tmpUUID = ''
                        for field in log['fields']:
                            if field['key'] == 'action':
                                if field['value'] == 'start':
                                    isStartLog = 1
                                elif field['value'] == 'end':
                                    isStartLog = 2
                            elif field['key'] == 'uuid':
                                tmpUUID = field['value']
                        if isStartLog == 0:
                            continue
                        if tmpUUID not in subCallDict:
                            subCallDict[tmpUUID] = [0, 0]
                        if isStartLog == 1:
                            subCallDict[tmpUUID][0] = log['timestamp']
                        elif isStartLog == 2:
                            subCallDict[tmpUUID][1] = log['timestamp']
                clientPodName = ''
                for tag in curSpan['process']['tags']:
                    if tag['key'] == 'pod.name':
                        clientPodName = tag['value']
                        break
                for subCallUUID in subCallDict:
                    if subCallUUID not in subCallUUID2Index:
                        continue
                    serverSpanIndex = subCallUUID2Index[subCallUUID]
                    serverSpan = totalSpans[serverSpanIndex]['_source']
                    if 'process' not in serverSpan:
                        continue
                    serverPodName = ''
                    for tag in serverSpan['process']['tags']:
                        if tag['key'] == 'pod.name':
                            serverPodName = tag['value']
                            break
                    serverDuration = serverSpan['duration']
                    callRelationship = f'{clientPodName}@{serverPodName}'
                    if callRelationship not in latencies:
                        latencies[callRelationship] = []
                    latency = (subCallDict[subCallUUID][1] - subCallDict[subCallUUID][0])-serverDuration
                    if latency < 0:
                        # print(f"延时为负数: {subCallDict[subCallUUID][1] - subCallDict[subCallUUID][0]} - {serverDuration}"
                        continue
                    latencies[callRelationship].append((subCallDict[subCallUUID][1] - subCallDict[subCallUUID][0])-serverDuration)
        return latencies

    def get_window_index(self, startTime, windowSize, timestamp):
        return floor((timestamp - startTime) // windowSize)

    def getPCData(self, startTime: int, endTime: int, windowSize: int, limit: int) -> Dict[str, List[int]]:
        '''
        获取指定时间段内各个pod调用的延时
        :param windowSize: 窗口大小
        :param startTime: 起始时间，秒级时间戳
        :param endTime: 结束时间，秒级时间戳
        :param limit: 查询过程中限制的spans数量上限, -1为不限制
        :return: Dict[调用关系, 延时列表]，其中调用关系为"clientPodName@serverPodName", 延时是微秒级别的
        '''
        totalSpans = self.getSpans(startTime, endTime, limit)
        spanID2Index = {}  # spanID到totalSpans的index的映射
        subCallUUID2Index = {}  # subCallUUID到totalSpans的index的映射
        # 后续窗口划分以ms为单位
        startTime *= 1000
        endTime *= 1000
        for index in range(len(totalSpans)):
            curSpan = totalSpans[index]['_source']
            spanID2Index[curSpan['spanID']] = index
            if 'tags' in curSpan:
                for tag in curSpan['tags']:
                    if tag['key'] == 'subCallUUID':
                        subCallUUID2Index[tag['value']] = index
                        break
        LOG.logger.info("开始计算各个pod调用的延时")
        pcData = {}
        for index in range(len(totalSpans)):
            curSpan = totalSpans[index]['_source']
            if 'logs' in curSpan:
                subCallDict = {}  # (server span的subCallUUID, client span记录的起始时间, client span记录的结束时间)
                for log in curSpan['logs']:
                    if 'fields' in log:
                        isStartLog = 0  # 0表示无效，1表示是start log，2表示是end log
                        tmpUUID = ''
                        for field in log['fields']:
                            if field['key'] == 'action':
                                if field['value'] == 'start':
                                    isStartLog = 1
                                elif field['value'] == 'end':
                                    isStartLog = 2
                            elif field['key'] == 'uuid':
                                tmpUUID = field['value']
                        if isStartLog == 0:
                            continue
                        if tmpUUID not in subCallDict:
                            subCallDict[tmpUUID] = [0, 0]
                        if isStartLog == 1:
                            subCallDict[tmpUUID][0] = log['timestamp']
                        elif isStartLog == 2:
                            subCallDict[tmpUUID][1] = log['timestamp']
                clientPodName = ''
                for tag in curSpan['process']['tags']:
                    if tag['key'] == 'pod.name':
                        clientPodName = tag['value']
                        break
                for subCallUUID in subCallDict:
                    if subCallUUID not in subCallUUID2Index:
                        continue
                    serverSpanIndex = subCallUUID2Index[subCallUUID]
                    serverSpan = totalSpans[serverSpanIndex]['_source']
                    if 'process' not in serverSpan:
                        continue
                    serverPodName = ''
                    for tag in serverSpan['process']['tags']:
                        if tag['key'] == 'pod.name':
                            serverPodName = tag['value']
                            break

                    startVal = floor(subCallDict[subCallUUID][0] / 1000)
                    endVal = floor(subCallDict[subCallUUID][1] / 1000)

                    startIdx = self.get_window_index(startTime, windowSize, startVal)
                    endIdx = self.get_window_index(startTime, windowSize, endVal)
                    if serverPodName not in pcData:
                        pcData[serverPodName] = {'divide': [0] * ((round(endTime) - round(startTime)) // windowSize)}
                    window_cnt = (endTime - startTime) / windowSize
                    if startIdx == endIdx:
                        pcData[serverPodName]['divide'][startIdx] += (endVal - startVal)
                    else:
                        cnt = endIdx - startIdx + 1
                        for i in range(round(cnt)):
                            if startIdx + i >= window_cnt:
                                break
                            if i == 0 or i == cnt - 1:
                                if i == 0:
                                    divTime = startTime + windowSize * (startIdx + i + 1)
                                    pcData[serverPodName]['divide'][startIdx] += (divTime - startVal)
                                if i == cnt - 1:
                                    divTime = startTime + windowSize * (startIdx + i)
                                    pcData[serverPodName]['divide'][endIdx] += (endVal - divTime)
                            else:
                                pcData[serverPodName]['divide'][startIdx + i] += windowSize
        return pcData


ELASTICSEARCH = ElasticSearchUtil(CONFIG["elasticSearch.host"], CONFIG["elasticSearch.port"])

