from ts.log.logger import Logger


class FileLogger(Logger):

    def __init__(self, filepath, logLevel=1):
        self.file = open(filepath, 'a')
        self.logLevel = logLevel

    def setLevel(self, logLevel):
        self.logLevel = logLevel

    def log(self, message, level, functionName=None):
        if self.logLevel >= level:
            if functionName is not None:
                self.file.write(functionName + ": " + message + "\n")
            else:
                self.file.write(message + "\n")

    def close(self):
        self.file.close()
