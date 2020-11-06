from ts.log.log import Log


class ConsoleLog(Log):

    def __init__(self, logLevel = 1):
        self.verboseLevel = logLevel

    def setLevel(self, logLevel):
        self.logLevel = logLevel

    def write(self, message, level):
        if self.logLevel >= level:
            print(message)

    def close(self):
        pass
