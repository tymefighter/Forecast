from ts.log.logger import Logger


class ConsoleLogger(Logger):

    def __init__(self, logLevel=1):
        self.logLevel = logLevel

    def setLevel(self, logLevel):
        self.logLevel = logLevel

    def log(self, message, level, functionName=None):
        if self.logLevel >= level:
            if functionName is not None:
                print(functionName + ": " + message, end='\n')
            else:
                print(message, end='\n')

    def close(self):
        pass
