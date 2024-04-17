import datetime
from typing import List

from entity.DetectionResult import DetectionResult
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class DetectionResultDAO:
    def insert(self, detectionResult: DetectionResult):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.add(detectionResult)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionResult.insert rollback")
            raise Exception

    def updateByTaskIdAndPodName(self, detectionResult: DetectionResult):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            oldResult = session.query(DetectionResult).filter(DetectionResult.taskId == detectionResult.taskId, DetectionResult.podName == detectionResult.podName).first()
            # 更新部分数据
            oldResult.abnormal = detectionResult.abnormal
            oldResult.maxScore = detectionResult.maxScore
            session.add(oldResult)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionResultDAO.update rollback")
            raise Exception

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
