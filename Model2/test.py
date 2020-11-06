import model

def testInitialize():
    pass

def testAttention():
    pass

def testPredictTimestep():
    pass

def testPredict():
    pass

def testRunGru():
    pass

def testBuildMemory():
    pass

def testSaveLoad():
    pass

def runTests():
    tests = [
        testInitialize, testAttention, testPredictTimestep, 
        testPredict, testRunGru, testBuildMemory, testSaveLoad
    ]

    for test in tests:
        test()

if __name__ == '__main__':
    runTests()