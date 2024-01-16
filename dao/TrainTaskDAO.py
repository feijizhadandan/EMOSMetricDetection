from entity.TrainTask import TrainTask
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class TrainTaskDAO:
    def insert(self, trainTask: TrainTask):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.add(trainTask)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("TrainTaskDAO.insert rollback")
            raise Exception

    def update(self, trainTask: TrainTask):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.merge(trainTask)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("TrainTaskDAO.update rollback")
            raise Exception

    def getAll(self):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(TrainTask).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("ModelMetadataDAO.insert rollback")
            raise Exception

    def getPage(self, page: int, pageSize: int):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(TrainTask).limit(pageSize).offset((page - 1) * pageSize).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("TrainTaskDAO.getPage rollback")
            raise Exception
