class UnivariateModel:

    def train(
            self,
            exogenousSeries,
            targetSeries,
            forecastHorizon,
            sequenceLength,
            modelSavePath=None
    ):
        pass

    def predict(
            self,
            exogenousSeries,
            targetSeries
    ):
        pass

    def save(self):
        pass

    def load(self):
        pass
