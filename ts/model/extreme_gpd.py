import tensorflow as tf
import numpy as np
import random

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

from ts.experimental import GeneralizedParetoDistribution, GpdEstimate, Pso


class ExtremeGpd:

    def __init__(
            self,
            threshold,
            lag,
            optimizerExtremeParam,
            optimizerNormalParam,
            optimizerEnsembleParam,
            numModelsEnsemble,
            negPosDataRatio
    ):

        # Extreme Model
        self.modelExtreme = ExtremeGpd.getExtremeModel(optimizerExtremeParam, lag)

        # Normal Model
        self.modelNormal = ExtremeGpd.getNormalModel(optimizerNormalParam, lag)

        # MLP Ensemble Classifier
        self.modelDetect = MlpEnsemble(
            lag, numModelsEnsemble, negPosDataRatio, optimizerEnsembleParam
        )

        # Save Parameters
        self.threshold = threshold
        self.lag = lag

        # GPD Distribution
        self.gpd = None

    def train(
            self,
            timeSeries,
            gpdParamRanges,
            numPsoParticles,
            numPsoIterations,
            numExtremeIterations,
            numNormalIterations,
            numEnsembleIterations
    ):
        trainSummary = dict()

        # Estimate GPD Parameters
        trainSummary['gpd-convergence'] = \
            self.estimateParams(timeSeries, gpdParamRanges, numPsoParticles, numPsoIterations)

        # Train Extreme Model
        trainSummary['loss-extreme'] = \
            self.trainExtremeModel(timeSeries, numExtremeIterations)

        # Train Normal Model
        trainSummary['loss-normal'] = \
            self.trainNormalModel(timeSeries, numNormalIterations)

        # Train Ensemble Model
        trainSummary['loss-ensemble'] = \
            self.trainEnsembleModel(timeSeries, numEnsembleIterations)

        return trainSummary

    def predict(self, timeSeries, getAllOutputs=False):

        # Build Input
        inputData = []
        for i in range(self.lag, timeSeries.shape[0] + 1):
            inputData.append(timeSeries[i - self.lag: i])

        inputData = np.array(inputData)

        # Predict Timestep is Extreme or Not using Ensemble
        isExtreme = self.modelDetect.predict(inputData)
        isExtreme = np.squeeze(isExtreme, axis=1)

        # Predict using Extreme Model
        predExtreme = self.gpd.computeQuantile(self.modelExtreme.predict(inputData)) \
            + self.threshold
        predExtreme = np.squeeze(predExtreme, axis=1)

        # Predict using Normal Model
        predNormal = self.modelNormal.predict(inputData)
        predNormal = np.squeeze(predNormal, axis=1)

        # Whenever Timestep is Extreme use Extreme Model, else use Normal
        # Model's prediction
        predOutputs = np.zeros(predNormal.shape)
        for i in range(predOutputs.shape[0]):
            predOutputs[i] = predExtreme[i] if isExtreme[i] else predNormal[i]

        if not getAllOutputs:
            return predOutputs
        else:
            return predOutputs, isExtreme, predNormal, predExtreme

    def estimateParams(
            self,
            timeSeries,
            gpdParamRanges,
            numPsoParticles,
            numPsoIterations
    ):

        # Compute Exceedances Series
        exceedSeries = timeSeries[timeSeries > self.threshold] - self.threshold

        # Compute GPD Parameters by performing ML Estimation using PSO
        params, maxLogLikelihood, maxLogLikelihoodValues = GpdEstimate.psoMethod(
            exceedSeries,
            Pso.computeInitialPos(gpdParamRanges, numPsoParticles),
            numIterations=numPsoIterations
        )

        # Create the GPD Distribution Object
        self.gpd = GeneralizedParetoDistribution(*params)

        return maxLogLikelihoodValues

    def trainExtremeModel(self, timeSeries, numExtremeIterations):

        inputData = []
        outputData = []

        for i in range(self.lag, timeSeries.shape[0]):

            if timeSeries[i] > self.threshold:
                inputData.append(timeSeries[i - self.lag: i])

                exceedance = timeSeries[i] - self.threshold
                outputData.append(self.gpd.cdf(exceedance))

        inputData = np.array(inputData)
        outputData = np.expand_dims(np.array(outputData), axis=1)

        history = self.modelExtreme.fit(
            inputData, outputData, epochs=numExtremeIterations,
            verbose=0
        )

        return history.history['loss']

    def trainNormalModel(self, timeSeries, numNormalIterations):

        inputData = []
        outputData = []

        for i in range(self.lag, timeSeries.shape[0]):

            if timeSeries[i] <= self.threshold:
                inputData.append(timeSeries[i - self.lag: i])
                outputData.append(timeSeries[i])

        inputData = np.array(inputData)
        outputData = np.expand_dims(np.array(outputData), axis=1)

        history = self.modelNormal.fit(
            inputData, outputData, epochs=numNormalIterations,
            verbose=0
        )

        return history.history['loss']

    def trainEnsembleModel(self, timeSeries, numEnsembleIterations):

        inputData = []
        outputData = []

        for i in range(self.lag, timeSeries.shape[0]):

            inputData.append(timeSeries[i - self.lag: i])

            if timeSeries[i] > self.threshold:
                outputData.append(1)

            else:
                outputData.append(0)

        inputData = np.array(inputData)
        outputData = np.expand_dims(np.array(outputData), axis=1)

        return self.modelDetect.train(inputData, outputData, numEnsembleIterations)

    @staticmethod
    def getExtremeModel(optimizerExtremeParam, lag):

        modelExtreme = Sequential([
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])

        modelExtreme.build(input_shape=(None, lag))
        modelExtreme.compile(
            optimizer=Adam(
                ExponentialDecay(
                    *optimizerExtremeParam
                )
            ),
            loss=tf.losses.MeanSquaredError()
        )

        return modelExtreme

    @staticmethod
    def getNormalModel(optimizerNormalParam, lag):

        modelNormal = Sequential([
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear'),
        ])

        modelNormal.build(input_shape=(None, lag))
        modelNormal.compile(
            optimizer=Adam(
                ExponentialDecay(
                    *optimizerNormalParam
                )
            ),
            loss=MeanSquaredError()
        )

        return modelNormal


class MlpEnsemble:

    def __init__(
            self,
            lag, numModels,
            negPosDataRatio, optimizerEnsembleParam
    ):
        self.models = [MlpEnsemble.getModel(lag) for _ in range(numModels)]
        self.negPosDataRatio = negPosDataRatio
        self.optimizerEnsembleParam = optimizerEnsembleParam

    def train(self, inputData, outputData, numEachModelEpochs):

        posData = []
        negData = []

        for i in range(inputData.shape[0]):

            if outputData[i, 0] == 1:
                posData.append((inputData[i], outputData[i]))

            else:
                negData.append((inputData[i], outputData[i]))

        assert len(posData) < len(negData)

        losses = np.zeros(numEachModelEpochs)
        num_neg_to_take = int(len(posData) * self.negPosDataRatio)

        for model in self.models:

            data = posData.copy()
            data.extend(random.sample(negData, k=num_neg_to_take))

            input_data, output_data = [], []
            for inp, out in data:
                input_data.append(inp)
                output_data.append(out)

            input_data = np.array(input_data)
            output_data = np.array(output_data)

            history = model.fit(
                input_data,
                output_data,
                epochs=numEachModelEpochs,
                verbose=0
            )

            losses += np.array(history.history['loss'])

        losses /= len(self.models)
        return losses

    def predict(self, inputData):

        numPos = np.zeros((inputData.shape[0], 1), dtype=np.int64)
        numNeg = np.zeros((inputData.shape[0], 1), dtype=np.int64)

        for model in self.models:
            model_out = model.predict(inputData)
            model_pred = (tf.sigmoid(model_out).numpy() > 0.5) \
                .astype(np.int64)

            numPos += model_pred
            numNeg += (1 - model_pred)

        pred = (numPos > numNeg).astype(np.int64)
        return pred

    @staticmethod
    def getModel(lag):

        model = Sequential([
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.build(input_shape=(None, lag))
        model.compile(
            optimizer=Adam(
                ExponentialDecay(
                    1e-3, 50, 0.9
                )
            ),
            loss=BinaryCrossentropy(from_logits=True)
        )

        return model
