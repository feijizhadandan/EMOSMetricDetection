from typing import Dict

'''
管理训练线程以及检测线程的单例
'''


class threadUtilClass:
    '''
    训练任务的线程句柄与状态集合, 只能有一个在线实时数据训练线程，其键为taskId, 值为{"threadRef":xxx, "status":xxx}
    "threadRef"为线程句柄
    "status"取值为running, stopping, stopped
    '''
    trainingThreadsMap: Dict[str, Dict] = {}

    '''
    检测任务的线程句柄与状态集合, 只能有一个在线数据检测线程，其键为taskId, 值为{"threadRef":xxx, "status":xxx}
    "threadRef"为线程句柄
    "status"取值为running, stopping, stopped
    '''
    detectionThreadsMap: Dict[str, Dict] = {}

    '''
    当前在线实时检测任务的taskId
    '''
    onlineDetectionTaskId = ""


threadUtil = threadUtilClass()
