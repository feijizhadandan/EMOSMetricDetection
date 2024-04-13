from sqlalchemy import Column, String, BigInteger

from entity.SQLAlchemyBase import SQLAlchemyBase


class RootCauseResult(SQLAlchemyBase):
    '''
    检测结果

    id: 数据库自增id

    traceId: traceId

    '''
    __tablename__ = 'metric_root_cause_result'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    taskId = Column(String)
    top_1 = Column(String)
    top_2 = Column(String)
    top_3 = Column(String)
    top_4 = Column(String)
    top_5 = Column(String)

    def as_dict(self):
        return {
            "id": self.id,
            "taskId": self.taskId,
            "top_1": self.top_1,
            "top_2": self.top_2,
            "top_3": self.top_3,
            "top_4": self.top_4,
            "top_5": self.top_5,
        }
