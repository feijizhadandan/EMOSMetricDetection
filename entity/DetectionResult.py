from sqlalchemy import Column, Integer, String, Boolean, BigInteger, Float, DateTime, func

from entity.SQLAlchemyBase import SQLAlchemyBase


class DetectionResult(SQLAlchemyBase):
    '''
    检测结果

    id: 数据库自增id

    traceId: traceId

    taskId: 任务id，uuid，记录这个是为了知道是哪次任务得出的这个检测结果

    abnormal: 是否异常，bool, 正常为0，异常为1

    loss: float, traceGRA算得的loss值

    threshold: float, 超过该值则认为是异常

    traceTime: int, trace的时间戳, ns级别
    '''
    __tablename__ = 'metric_detection_result'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    traceId = Column(String)
    taskId = Column(String)
    abnormal = Column(Boolean)
    loss = Column(Float)
    threshold = Column(Float)
    traceTime = Column(BigInteger)

    def as_dict(self):
        return {
            "id": self.id,
            "traceId": self.traceId,
            "taskId": self.taskId,
            "abnormal": self.abnormal,
            "loss": self.loss,
            "threshold": self.threshold,
            "traceTime": self.traceTime
        }
