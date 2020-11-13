from ts.log.logger import DEFAULT_LOG_PATH
from ts.log.file import FileLogger


class GlobalLogger:
    """
    Global Logger Class which provides access to a single
    global file logger
    """

    """ Static logger object """
    logger = FileLogger(DEFAULT_LOG_PATH)

    @staticmethod
    def getLogger():
        """
        Get the Global Logger

        :return: The reference to the global logger
        """
        return GlobalLogger.logger

    @staticmethod
    def setPath(logPath):
        """
        Update the global logger to write to the file at the
        filepath provided

        :param logPath: Path to the file to which the global
        logger should log to
        :return: None
        """
        GlobalLogger.logger.close()
        GlobalLogger.logger = FileLogger(logPath)
