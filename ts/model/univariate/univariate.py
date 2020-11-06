class UnivariateModel:

    def train(
            self,
            exogenousSeries,
            targetSeries,
            forecastHorizon,
            sequenceLength,
            modelSavePath=None,
            verbose=1
    ):
        pass

    def predict(
            self,
            exogenousSeries,
            targetSeries
    ):
        pass

    def save(
            self,
            modelSavePath
    ):
        pass

    def load(
            self,
            modelLoadPath
    ):
        pass
