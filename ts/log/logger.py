class Logger:
    """Base Logger Class"""

    def setLevel(self, logLevel):
        """Set Logging Level, high level means more information"""

        pass

    def log(self, message, level, functionName=None):
        """
        Log the message at the provided logging level

        :param message: Message to be logged
        :param level: Logging Level of the message
        :param functionName: Function from which the log function was called
        """

        pass

    def close(self):
        """Close the Logger"""

        pass
