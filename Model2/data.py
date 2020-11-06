class Data:

    def getOrGenerateData(
        dataPath = None,
        n = 1000,
        inputDimension = 1,
        horizon = 1,
        extremeValues = False,
        longTerm = False
    ):
        if extremeValues:
            noiseGenFunc = np.random.gumbel
            noiseGenParams = (100., 10.0, (inputDimension,))
        else:
            noiseGenFunc = np.random.normal
            noiseGenParams = (10.0, 1.0, (inputDimension,))

        if longTerm:
            P = 50
            Q = 50
        else:
            P = 5
            Q = 5

        obsCoef, noiseCoef = generateCoeff(P, Q)

        X = generateArma(
            n + h,
            obsCoef,
            noiseCoef,
            noiseGenFunc,
            noiseGenParams
        )
        y = X[h :, 0]

        return X[ : n], y
        
    def generateCoeff(P, Q):
        obsCoef = np.concatenate([
            np.random.uniform(-0.1, 0, size = P // 2),
            np.random.uniform(0, 0.1, size = P // 2)
        ])

        noiseCoef = np.concatenate([
            np.random.uniform(-0.01, 0, size = Q // 2),
            np.random.uniform(0, 0.01, size = Q // 2)
        ])

        return obsCoef, noiseCoef

    def generateArma(
        n,
        obsCoef,
        noiseCoef,
        noiseGenFunc,
        noiseGenParams,
        obsFunc = None,
        noiseFunc = None
    ):

        p = len(obsCoef)
        q = len(noiseCoef)    
        
        x = [None] * n
        eps = [None] * n

        for t in range(n):

            obsVal = 0
            for i in range(min(t, p)):
                obsVal += obsCoef[i] * x[t - i - 1]
            
            if obsFunc is not None:
                obsVal = obsFunc(obsVal)
            x[t] += obsVal
            
            noiseVal = 0
            for j in range(min(t, q)):
                noiseVal += noiseCoef[j] * eps[t - j - 1]

            if noiseFunc is not None:
                noiseVal = noiseFunc(noiseVal)
            x[t] += noiseVal

            eps[t] = noiseGenFunc(*noiseGenParams)
            x[t] += eps[t]

        return np.array(x)