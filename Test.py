import uuid
from algorithm.metric.train import onlineTrain
from entity.TrainTask import TrainTask
from utils.prometheusUtil import PROMETHEUS

def process_time_windows(startTime, endTime, windowSize, arr):
    res = {}

    window_cnt = (endTime - startTime) / windowSize

    def get_window_index(timestamp):
        return (timestamp - startTime) // windowSize

    for obj in arr:
        startVal = obj['start']
        endVal = obj['end']
        startIdx = get_window_index(obj['start'])
        endIdx = get_window_index(obj['end'])
        serverPod = obj['serverPod']
        if serverPod not in res:
            res[serverPod] = {'divide': [0] * ((endTime - startTime) // windowSize)}
        if startIdx == endIdx:
            res[serverPod]['divide'][startIdx] += (endVal - startVal)
        else:
            cnt = endIdx - startIdx + 1
            for i in range(cnt):
                if startIdx + i >= window_cnt:
                    break
                if i == 0 or i == cnt - 1:
                    if i == 0:
                        divTime = startTime + windowSize * (startIdx + i + 1)
                        res[serverPod]['divide'][startIdx] += (divTime - startVal)
                    if i == cnt - 1:
                        divTime = startTime + windowSize * (startIdx + i)
                        res[serverPod]['divide'][endIdx] += (endVal - divTime)
                else:
                    res[serverPod]['divide'][startIdx + i] += windowSize

    return res

if __name__ == "__main__":
    # 示例用法
    startTime = 200
    endTime = 300
    windowSize = 10
    arr = [{'serverPod': 'A', 'start': 199, 'end': 220}]
    result = process_time_windows(startTime, endTime, windowSize, arr)
    print(result)

