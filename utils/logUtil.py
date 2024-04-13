import logging

from .configUtil import CONFIG

'''
系统输出日志的单例类
'''
class LogUtil:
    _logger = None

    def __new__(cls):
        if cls._logger is None:
            cls._logger = super(LogUtil, cls).__new__(cls)
            cls._logger.setup_logger()
        return cls._logger

    def setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # 创建一个日志文件，将日志记录到文件中
        file_handler = logging.FileHandler(CONFIG['log.path'], mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # 创建一个控制台处理器，将日志同时输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

# 使用单例模式创建日志记录器, 使用的时候LOG.logger.info()这样用
LOG = LogUtil()
