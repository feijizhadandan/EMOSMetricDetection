from sqlalchemy import Column, Integer, String, DateTime, BigInteger, JSON
from sqlalchemy.sql import func

from entity.SQLAlchemyBase import SQLAlchemyBase
from utils.timeFormatUtil import TIMEFORMAT


class ModelMetadata(SQLAlchemyBase):
    '''
    检测任务

    id: 数据库自增id

    modelName: 模型名, str, 不要超过64个字符

    createTime: 模型创建时间, DateTime

    meta: 模型的参数, JSON
    '''
    __tablename__ = 'model_metadata'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    modelName = Column(String)
    mark = Column(Integer)
    createTime = Column(DateTime(timezone=True), default=func.now())
    meta = Column(JSON)

    def as_dict(self):
       return {
           "id": str(self.id),
           "modelName": str(self.modelName),
           "mark": str(self.mark),
           "createTime": TIMEFORMAT.addLocalTzinfo(self.createTime),
           "meta": str(self.meta)
       }
