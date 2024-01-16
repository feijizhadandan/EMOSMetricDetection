from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session

from entity.SQLAlchemyBase import SQLAlchemyBase
from utils.configUtil import CONFIG


class MysqlUtil:
    def __init__(self, host, port, username, password, dbname):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.dbname = dbname
        self.engine = create_engine(
            f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.dbname}",
            pool_size=100,  # 连接池中的连接数
            max_overflow=0  # 超过连接池大小外最多创建的连接
        )
        self.scopedSessionFactory = scoped_session(sessionmaker(bind=self.engine))

    def testConnection(self):
        try:
            with self.engine.connect() as connection:
                print("连接mysql成功")
        except:
            print("连接mysql失败")


MYSQL = MysqlUtil(
    CONFIG['mysql.host'],
    CONFIG['mysql.port'],
    CONFIG['mysql.username'],
    CONFIG['mysql.password'],
    CONFIG['mysql.dbname']
)
