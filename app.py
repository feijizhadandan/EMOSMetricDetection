import datetime
import threading
import uuid

from flask import Flask, request, jsonify

from algorithm.metric.detect import detectOnline
from algorithm.metric.train import onlineTrain
from entity.DetectionTask import DetectionTask
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
        # TODO: 将训练线程信息保存到数据库
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
        threadUtil.onlineDetectionTaskId = detectionTask.taskId
        threadUtil.detectionThreadsMap[detectionTask.taskId] = {"threadRef": onlineThread, "status": "running"}
        # TODO: 将检测线程信息保存到数据库
        onlineThread.start()
        return jsonify({
            "success": True,
            "msg": "成功创建在线实时检测任务",
            "taskId": detectionTask.taskId
        })
    except:
        return jsonify({"success": False, "msg": "创建在线实时检测任务失败", "taskId": ""})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8888)
