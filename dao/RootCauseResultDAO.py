import datetime
from typing import List

from entity.RootCauseResult import RootCauseResult
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class RootCauseResultDAO:
    def insert(self, rootCauseResult: RootCauseResult):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.add(rootCauseResult)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("RootCauseResult.insert rollback")
            raise Exception

    def bulkInsert(self, rootCauseResultList: List[RootCauseResult]):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.bulk_save_objects(rootCauseResultList)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("RootCauseResultDao.bulkInsert rollback")
            raise Exception

    def getByTaskId(self, taskId: str):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(RootCauseResult).filter(RootCauseResult.taskId == taskId).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("RootCauseResultDao.getByTaskId rollback")
            raise Exception
