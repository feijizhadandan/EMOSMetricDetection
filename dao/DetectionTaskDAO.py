from entity.DetectionTask import DetectionTask
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class DetectionTaskDAO:
    def insert(self, detectionTask: DetectionTask):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.add(detectionTask)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("TrainTaskDAO.insert rollback")
            raise Exception

    def update(self, detectionTask: DetectionTask):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.merge(detectionTask)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionTaskDAO.update rollback")
            raise Exception

    def getAll(self):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionTask).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionTaskDAO.getall rollback")
            raise Exception

    def getByTaskId(self, taskId):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionTask).filter(DetectionTask.taskId == taskId).first()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionTaskDAO.getByTaskId rollback")
            raise Exception

    def getPage(self, page: int, pageSize: int):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(DetectionTask).limit(pageSize).offset((page - 1) * pageSize).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("DetectionTaskDAO.getPage rollback")
            raise Exception
