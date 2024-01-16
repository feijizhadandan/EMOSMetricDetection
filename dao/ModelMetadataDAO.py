from entity.ModelMetadata import ModelMetadata
from utils.logUtil import LOG
from utils.mysqlUtil import MYSQL


class ModelMetadataDAO:
    def insert(self, modelMetadata: ModelMetadata):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            session.add(modelMetadata)
            session.commit()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("ModelMetadataDAO.insert rollback")
            raise Exception

    def getAll(self):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(ModelMetadata).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("ModelMetadataDAO.insert rollback")
            raise Exception

    def getPaged(self, page: int, pageSize: int):
        session = None
        try:
            session = MYSQL.scopedSessionFactory()
            return session.query(ModelMetadata).limit(pageSize).offset((page - 1) * pageSize).all()
        except Exception:
            if session:
                session.rollback()
            LOG.logger.exception("ModelMetadataDAO.insert rollback")
            raise Exception
