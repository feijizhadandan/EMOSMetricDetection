import datetime
from typing import List

from entity.DetectionScoreResult import DetectionScoreResult
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class DetectionScoreResultDAO:
    def bulkInsert(self, detectionScoreResultList: List[DetectionScoreResult]):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.bulk_save_objects(detectionScoreResultList)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionScoreResultDAO.bulkInsert rollback")
            raise Exception

    def getByTaskId(self, taskId: str):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionScoreResult).filter(DetectionScoreResult.taskId == taskId).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionScoreResultDAO.getByTaskId rollback")
            raise Exception
