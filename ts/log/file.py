from ts.log.logger import Logger


class FileLogger(Logger):

    def __init__(self, filepath, logLevel=1):
        self.file = open(filepath, 'a')
        self.logLevel = logLevel

    def setLevel(self, logLevel):
        self.logLevel = logLevel

    def write(self, message, level):
        if self.logLevel >= level:
            self.file.write(message)

    def close(self):
        self.file.close()
