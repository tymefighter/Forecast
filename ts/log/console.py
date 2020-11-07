from ts.log.logger import Logger


class ConsoleLogger(Logger):
    """Console Logger"""

    def __init__(self, logLevel=1):
        """
        Initialize Console Logger

        :param logLevel: Logging Level on which the Logger operates on
        """

        self.logLevel = logLevel

    def setLevel(self, logLevel):
        """Set Logging Level, high level means more information"""

        self.logLevel = logLevel

    def log(self, message, level, functionName=None):
        """
        Log the message at the provided logging level to the console

        :param message: Message to be logged
        :param level: Logging Level of the message
        :param functionName: Function from which the log function was called
        """

        if self.logLevel >= level:
            if functionName is not None:
                print(functionName + ": " + message, end='\n')
            else:
                print(message, end='\n')

    def close(self):
        """
        Close the Logger

        Does nothing since there is not resource to be released
        """
        pass
