from sqlalchemy import Column, Integer, String, DateTime, BigInteger
from sqlalchemy.sql import func

from entity.SQLAlchemyBase import SQLAlchemyBase
from utils.timeFormatUtil import TIMEFORMAT


class DetectionTask(SQLAlchemyBase):
    '''
    检测任务

    id: 数据库自增id

    taskId: 任务id，uuid

    createTime: 任务创建时间, DateTime

    modelName: 检测任务用到的模型名, str, 不要超过64个字符

    onlineTask: 是否为在线实时检测, 0不是，1是

    startTime: 检测数据起始时间，DateTime，如果onlineTask为1.则此项无意义

    endTime: 检测数据结束时间，DateTime，如果onlineTask为1.则此项无意义

    windowSize: 检测窗口长度(秒级时间戳)

    status: 任务运行状态, str, 取值为("running", "stopped")
    '''
    __tablename__ = 'tracegra_detection_task'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    taskId = Column(String)
    createTime = Column(DateTime(timezone=True), default=func.now())
    modelName = Column(String)
    onlineTask = Column(Integer)
    startTime = Column(DateTime(timezone=True), default=func.now())
    endTime = Column(DateTime(timezone=True), default=func.now())
    windowSize = Column(Integer)
    status = Column(String)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "createTime": TIMEFORMAT.addLocalTzinfo(self.createTime),
            "modelName": self.modelName,
            "onlineTask": self.onlineTask,
            "startTime": TIMEFORMAT.addLocalTzinfo(self.startTime),
            "endTime": TIMEFORMAT.addLocalTzinfo(self.endTime),
            "windowSize": self.windowSize,
            "status": self.status
        }
