from ts.log.logger import Logger


class FileLogger(Logger):
    """File Logger"""

    def __init__(self, filepath, logLevel=1):
        """
        Initialize File Logger

        :param filepath: Path of the Log File
        :param logLevel: Logging Level on which the Logger operates on
        """

        self.file = open(filepath, 'a')
        self.logLevel = logLevel

    def setLevel(self, logLevel):
        """Set Logging Level, high level means more information"""

        self.logLevel = logLevel

    def log(self, message, level, functionName=None):
        """
        Log the message at the provided logging level to the file

        :param message: Message to be logged
        :param level: Logging Level of the message
        :param functionName: Function from which the log function was called
        """

        if self.logLevel >= level:
            if functionName is not None:
                self.file.write(functionName + ": " + message + "\n")
            else:
                self.file.write(message + "\n")

    def close(self):
        """
        Close the Logger

        It closes the file resource
        """

        self.file.close()
