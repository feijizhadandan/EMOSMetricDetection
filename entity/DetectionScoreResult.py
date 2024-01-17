from sqlalchemy import Column, Integer, String, Boolean, BigInteger, Float, DateTime, func, Double

from entity.SQLAlchemyBase import SQLAlchemyBase


class DetectionScoreResult(SQLAlchemyBase):
    """

    """
    __tablename__ = 'detection_score_record'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    taskId = Column(String)
    podName = Column(String)
    score = Column(Double)
    startTime = Column(DateTime)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "podName": self.podName,
            "score": self.score,
            "startTime": self.startTime
        }
