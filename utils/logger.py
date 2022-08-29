import logging
import os
import sys
from datetime import datetime

from utils.singleton import singleton


class LogLevelFilter:
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level

@singleton
class Logger:
    def __init__(self) -> None:
        self.__name = 'bmd-estimation'
        self.__date_format = "%Y-%m-%d %H:%M:%S"
        self.__log_format = "%(asctime)s - %(levelname)s - %(message)s"
        self.__path_log = 'logs'
        self.__logger = None
        self.__start()

    def __start(self):
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(logging.INFO)

        date_time = datetime.now().strftime(self.__date_format)
        formatter = logging.Formatter(self.__log_format)

        if not os.path.exists(self.__path_log):
            os.makedirs(self.__path_log)

        file_handler = logging.FileHandler(os.path.join(self.__path_log, '{}_{}.log').format(self.__name, date_time), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(LogLevelFilter(logging.INFO))
        self.__logger.addHandler(stream_handler)

        stream_error_handler = logging.StreamHandler(sys.stderr)
        stream_error_handler.setLevel(logging.WARNING)
        stream_error_handler.setFormatter(formatter)
        self.__logger.addHandler(stream_error_handler)

    def info(self, message: str) -> None:
        self.__logger.info(message)

    def warning(self, message: str) -> None:
        self.__logger.warning(message)

    def error(self, message: str) -> None:
        self.__logger.error(message)
