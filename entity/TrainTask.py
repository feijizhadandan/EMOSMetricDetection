from sqlalchemy import Column, Integer, String, DateTime, BigInteger
from sqlalchemy.sql import func

from entity.SQLAlchemyBase import SQLAlchemyBase
from utils.timeFormatUtil import TIMEFORMAT


class TrainTask(SQLAlchemyBase):
    '''
    训练任务

    id: 数据库自增id

    taskId: 任务id，uuid

    createTime: 任务创建时间, DateTime

    startTime: 训练集起始时间，DateTime

    endTime: 训练集结束时间，DateTime

    onlineTask: 是否为在线实时检测, 0不是，1是

    modelName: 此次训练所得的模型名字, str,不要超过64个字符

    status: 任务运行状态, str, 取值为("running", "stopped")

    progress: 任务进度, int, 百分比
    '''
    __tablename__ = 'metric_train_task'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    taskId = Column(String)
    createTime = Column(DateTime(timezone=True), default=func.now())
    startTime = Column(DateTime(timezone=True), default=func.now())
    endTime = Column(DateTime(timezone=True), default=func.now())
    onlineTask = Column(Integer)
    modelName = Column(String)
    status = Column(String)
    progress = Column(Integer)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "createTime": TIMEFORMAT.addLocalTzinfo(self.createTime),
            "startTime": TIMEFORMAT.addLocalTzinfo(self.startTime),
            "endTime": TIMEFORMAT.addLocalTzinfo(self.endTime),
            "onlineTask": self.onlineTask,
            "modelName": self.modelName,
            "status": self.status,
            "progress": self.progress
        }
