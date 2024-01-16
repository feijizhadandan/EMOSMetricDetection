import datetime
from typing import List

from entity.DetectionResult import DetectionResult
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class DetectionResultDAO:
    def bulkInsert(self, detectionResultList: List[DetectionResult]):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.bulk_save_objects(detectionResultList)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionResultDAO.bulkInsert rollback")
            raise Exception

    def getByTaskId(self, taskId: str):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionResult).filter(DetectionResult.taskId == taskId).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionResultDAO.getByTaskId rollback")
            raise Exception

    def getByTaskIdAndTraceTimeBetween(self, taskId: str, startTime: int, endTime: int):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionResult).filter(DetectionResult.taskId == taskId,
                                                         DetectionResult.traceTime.between(startTime, endTime)).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionResultDAO.getByTaskId rollback")
            raise Exception
