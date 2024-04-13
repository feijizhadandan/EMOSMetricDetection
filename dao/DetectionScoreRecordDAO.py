import datetime
from typing import List

from entity.DetectionScoreRecord import DetectionScoreRecord
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class DetectionScoreRecordDAO:
    def bulkInsert(self, detectionScoreRecordList: List[DetectionScoreRecord]):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.bulk_save_objects(detectionScoreRecordList)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionScoreRecordDAO.bulkInsert rollback")
            raise Exception

    def getByTaskId(self, taskId: str):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionScoreRecord).filter(DetectionScoreRecord.taskId == taskId).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionScoreRecordDAO.getByTaskId rollback")
            raise Exception
