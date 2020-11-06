from ts.log.logger import Logger


class ConsoleLogger(Logger):

    def __init__(self, logLevel=1):
        self.logLevel = logLevel

    def setLevel(self, logLevel):
        self.logLevel = logLevel

    def write(self, message, level):
        if self.logLevel >= level:
            print(message)

    def close(self):
        pass
