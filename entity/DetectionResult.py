from sqlalchemy import Column, Integer, String, Boolean, BigInteger, Float, DateTime, func, Double

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
    taskId = Column(String)
    podName = Column(String)
    abnormal = Column(Boolean)
    maxScore = Column(Double)
    threshold = Column(Double)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "podName": self.podName,
            "abnormal": self.abnormal,
            "maxScore": self.maxScore,
            "threshold": self.threshold,
        }
