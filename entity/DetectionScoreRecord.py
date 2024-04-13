from sqlalchemy import Column, Integer, String, Boolean, BigInteger, Float, DateTime, func, Double

from entity.SQLAlchemyBase import SQLAlchemyBase


class DetectionScoreRecord(SQLAlchemyBase):
    """

    """
    __tablename__ = 'metric_detection_score_record'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    taskId = Column(String)
    podName = Column(String)
    score = Column(Double)
    startTime = Column(DateTime)
    normalCnt = Column(Integer)
    abnormalCnt = Column(Integer)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "podName": self.podName,
            "score": self.score,
            "startTime": self.startTime,
            "normalCnt": self.normalCnt,
            "abnormalCnt": self.abnormalCnt
        }
