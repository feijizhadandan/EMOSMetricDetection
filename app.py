import datetime
import os
import threading
import uuid
from typing import List

from flask import Flask, request, jsonify

from algorithm.metric.detect import detectOnline
from algorithm.metric.train import onlineTrain
from dao.DetectionResultDAO import DetectionResultDAO
from dao.DetectionScoreResultDAO import DetectionScoreResultDAO
from dao.DetectionTaskDAO import DetectionTaskDAO
from dao.ModelMetadataDAO import ModelMetadataDAO
from dao.TrainTaskDAO import TrainTaskDAO
from entity.DetectionTask import DetectionTask
from entity.ModelMetadata import ModelMetadata
from entity.TrainTask import TrainTask
from utils.threadUtil import threadUtil

app = Flask(__name__)


@app.route('/train', methods=['POST'])
def train():
    requestParam = request.get_json()
    try:
        startTime = datetime.datetime.fromisoformat(requestParam["startTime"])
        endTime = datetime.datetime.fromisoformat(requestParam["endTime"])
        currentTime = datetime.datetime.now()
        trainTask = TrainTask(
            taskId=str(uuid.uuid4()),
            createTime=currentTime,
            startTime=startTime,
            endTime=endTime,
            onlineTask=0,
            modelName=requestParam['modelName'],
            status="running",
            progress=0
        )
        trainingThread = threading.Thread(
            target=onlineTrain,
            args=(trainTask.taskId, requestParam["trainParam"], trainTask)
        )
        TrainTaskDAO().insert(trainTask=trainTask)
        # 将任务taskId作为线程name
        threadUtil.trainingThreadsMap[trainTask.taskId] = {"threadRef": trainingThread, "status": "running"}
        trainingThread.start()  # 启动训练线程
        return jsonify({"success": True, "msg": "成功创建训练任务", "taskId": trainTask.taskId})
    except:
        return jsonify({"success": False, "msg": "创建训练任务失败", "taskId": ""})


@app.route('/detect', methods=['POST'])
def detect():
    requestParam = request.get_json()
    try:
        startTime = datetime.datetime.fromisoformat(requestParam["startTime"])
        endTime = datetime.datetime.fromisoformat(requestParam["endTime"])
        curDateTime = datetime.datetime.now()
        detectionTask = DetectionTask(
            taskId=str(uuid.uuid4()),
            modelName=requestParam["modelName"],
            onlineTask=0,
            createTime=curDateTime,
            startTime=startTime,
            endTime=endTime,
            status="running"
        )
        onlineThread = threading.Thread(
            target=detectOnline,
            args=(detectionTask.taskId, requestParam["detectionParam"], detectionTask),
            daemon=True
        )
        # TODO 线程问题，为什么这个不能在insert之后执行
        onlineThread.start()
        threadUtil.onlineDetectionTaskId = detectionTask.taskId
        threadUtil.detectionThreadsMap[detectionTask.taskId] = {"threadRef": onlineThread, "status": "running"}
        DetectionTaskDAO().insert(detectionTask=detectionTask)
        return jsonify({
            "success": True,
            "msg": "成功创建在线实时检测任务",
            "taskId": detectionTask.taskId
        })
    except:
        return jsonify({"success": False, "msg": "创建在线实时检测任务失败", "taskId": ""})


@app.route("/listModel", methods=["GET"])
def listModel():
    modelMetadataList: List[ModelMetadata] = ModelMetadataDAO().getAll()
    filenameSet = set([fname for fname in os.listdir("./detectionmodels")])
    res = []
    for modelMetadata in modelMetadataList:
        if f"{modelMetadata.modelName}.pth" in filenameSet and modelMetadata.mark == 1:
            res.append(modelMetadata)
    return jsonify([modelMetadata.as_dict() for modelMetadata in res])


@app.route("/listTrainTask", methods=["GET"])
def listTrainTask():
    trainTaskList = TrainTaskDAO().getAll()
    return jsonify([trainTask.as_dict() for trainTask in trainTaskList])


@app.route("/listDetectionTask", methods=["GET"])
def listDetectionTask():
    detectionTaskList = DetectionTaskDAO().getAll()
    return jsonify([detectionTask.as_dict() for detectionTask in detectionTaskList])


@app.route("/getDetectionTaskByTaskId/<task_id>", methods=["GET"])
def getDetectionTaskByTaskId(task_id):
    # 获取曲线数据
    scoreResultList = DetectionScoreResultDAO().getByTaskId(task_id)
    scoreResultJson = [scoreResult.as_dict() for scoreResult in scoreResultList]
    # 获取检测结果
    detectionResultList = DetectionResultDAO().getByTaskId(task_id)
    detectionResultJson = [detectionResult.as_dict() for detectionResult in detectionResultList]
    # 获取检测模型
    metricDetectionTask = DetectionTaskDAO().getByTaskId(task_id)
    modelName = metricDetectionTask.modelName
    modelMeta = ModelMetadataDAO().getByModelName(modelName)
    modelMetaJson = modelMeta.as_dict()
    return jsonify({"scoreResultJson": scoreResultJson, "detectionResult": detectionResultJson, "modelMeta": modelMetaJson})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8888)
