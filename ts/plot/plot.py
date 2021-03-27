import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Plot:
    """Static Plotting Class"""

    @staticmethod
    def plotDataCols(
            X,
            colNames=None,
            title='Data Plot',
            savePath=None,
            saveOnly=False
    ):
        """
        Plots each column of X as a time series in a subplot

        :param X: Matrix with each column as a time series, it is a numpy array
        of shape (T, d) or a matrix of shape (T,) i.e. it has only one column
        :param colNames: Names of each column. If it is not None, then this would
        be present as the subplot title for that column
        :param title: Title of the entire plot
        :param savePath: If None, then does not save the plot, else saves the plot
        at the location provided
        :param saveOnly: Can be set to True only if savePath is not None. If True,
        then only saves the plot i.e. it does not plot it, If False, then it does
        will plot the figure
        :return: None
        """

        assert (savePath is None or (savePath is not None and saveOnly))

        if len(X.shape) == 1:
            d = 1
        else:
            d = X.shape[1]

        fig, axes = plt.subplots(d)
        fig.tight_layout()

        if d == 1:
            axes.plot(np.squeeze(X))

        else:
            for dim in range(d):
                ax = axes[dim]

                ax.plot(X[:, dim])
                if colNames is not None:
                    ax.set_title(colNames[dim])

        fig.suptitle(title)

        if savePath is not None:
            plt.savefig(savePath)

        if not saveOnly:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plotLoss(
            loss,
            title='Loss Plot',
            xlabel='Iterations',
            ylabel='Loss',
            savePath=None,
            saveOnly=False
    ):
        """
        Plots Loss vs Iterations Curve

        :param loss: 1D List or numpy array of shape (numIters,) of loss values
        which are to be plotted
        :param title: Title of the plot
        :param xlabel: x-axis Label of the plot
        :param ylabel: y-axis Label of the plot
        :param savePath: If None, then does not save the plot, else saves the plot
        at the location provided
        :param saveOnly: Can be set to True only if savePath is not None. If True,
        then only saves the plot i.e. it does not plot it, If False, then it does
        will plot the figure
        :return: None
        """

        assert (savePath is None or (savePath is not None and saveOnly))

        plt.plot(loss)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if savePath is not None:
            plt.savefig(savePath)

        if not saveOnly:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plotPredTrue(
            pred,
            target,
            title='Pred and True',
            xlabel='Timestep',
            ylabel='Value',
            savePath=None,
            saveOnly=False
    ):
        """
        Plot Predictions and True Targets

        :param pred: Univariate Predictions, numpy array of shape (n,)
        :param target: Univariate True Targets, numpy array of shape (n,)
        :param title: Title of the plot
        :param xlabel: x-axis Label of the plot
        :param ylabel: y-axis Label of the plot
        :param savePath: If None, then does not save the plot, else saves the plot
        at the location provided
        :param saveOnly: Can be set to True only if savePath is not None. If True,
        then only saves the plot i.e. it does not plot it, If False, then it does
        will plot the figure
        :return: None
        """

        assert pred.shape == target.shape
        assert savePath is None or (savePath is not None and saveOnly)

        plt.plot(target, 'r', label='true')
        plt.plot(pred, 'b', label='pred')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()

        if savePath is not None:
            plt.savefig(savePath)

        if not saveOnly:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plotMultivariatePredTrue(
            pred,
            target,
            numRows=None,
            seriesNames=None,
            savePath=None,
            saveOnly=False
    ):
        """
        Plot Multivariate Predictions and True Targets

        :param pred: Multivariate Predictions, numpy array of shape (n, d)
        :param target: Multivariate True Targets, numpy array of shape (n, d)
        :param numRows: Number of rows of the subplot, if None, then each
        series will be plotted in a different row. If not None, then it should
        divide d (number of time series)
        :param seriesNames: Name of each of the d time series, if None then
        no name is given to the subplots of each series
        :param savePath: If None, then does not save the plot, else saves the plot
        at the location provided
        :param saveOnly: Can be set to True only if savePath is not None. If True,
        then only saves the plot i.e. it does not plot it, If False, then it does
        will plot the figure
        :return: None
        """

        assert pred.shape == target.shape
        assert len(pred.shape) == 2
        assert savePath is None or (savePath is not None and saveOnly)

        d = pred.shape[1]

        numRows = d if numRows is None else numRows
        assert d % numRows == 0
        numCols = d // numRows

        fig, axes = plt.subplots(
            nrows=numRows,
            ncols=numCols,
            figsize=(5 * numCols, 4 * numRows)
        )
        fig.tight_layout(pad=4.0)

        idx = 0
        for i in range(numRows):
            for j in range(numCols):
                if numRows == 1 and numCols == 1:
                    axis = axes
                elif numRows == 1:
                    axis = axes[j]
                elif numCols == 1:
                    axis = axes[i]
                else:
                    axis = axes[i, j]

                seriesName = seriesNames[idx] if seriesNames is not None \
                    else f'series {idx}'

                axis.plot(target, 'r', label='true')
                axis.plot(pred, 'b', label='pred')
                axis.set_title(f'{seriesName} vs. timesteps')

                axis.set_xlabel('timesteps')
                axis.set_ylabel(seriesName)

                idx += 1

        if savePath is not None:
            plt.savefig(savePath)

        if not saveOnly:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def plotTable(header, headerLoc, data, decimalPrecision='2'):
        """
        Plot a table of the given data

        :param header: the table header, a 1D list of strings
        :param headerLoc: location of the table header - values: 'top' or 'left'
        :param data: the data to be put in the table, numpy array of shape (n, m).
        If headerLoc is 'top', then number of columns in 'data' must match length
        of 'header', else number of rows in 'data' must match length of 'header'
        :param decimalPrecision: precision of the data
        :return: None
        """

        assert (headerLoc == 'top' and len(header) == data.shape[1]) \
            or (headerLoc == 'left' and len(header) == data.shape[0])

        (n, m) = data.shape
        cellText = [
            [f'{float(data[i, j]): .{decimalPrecision}f}' for j in range(m)]
            for i in range(n)
        ]

        axes = plt.gca()
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        plt.box(on=None)

        if headerLoc == 'top':
            table = plt.table(
                cellText=cellText, loc='center', cellLoc='center',
                colLabels=header, colColours=['#09ECFF' for i in range(m)]
            )
        else:
            table = plt.table(
                cellText=cellText, loc='center', cellLoc='center',
                rowLabels=header, rowColours=['#DDFF09' for i in range(n)]
            )

        table.set_fontsize(14)
        table.scale(3, 2.5)

        plt.show()

    @staticmethod
    def plotConfusionMatrixAsymmetric(yTrue, yPred, trueWidth, predWidth, precision=2):
        """

        :param yTrue:
        :param yPred:
        :param trueWidth:
        :param predWidth:
        :param precision:
        :return:
        """

        assert yTrue.shape == yPred.shape and len(yTrue.shape) == 1

        minTrue, maxTrue = yTrue.min(), yTrue.max()
        minPred, maxPred = yPred.min(), yPred.max()

        trueValues = np.arange(minTrue, maxTrue + trueWidth, trueWidth)
        numTrue = trueValues.shape[0]

        predValues = np.arange(minPred, maxPred + predWidth, predWidth)
        numPred = predValues.shape[0]

        confusionMatrix = np.zeros((numTrue - 1, numPred - 1))

        n = yTrue.shape[0]
        for i in range(n):

            trueIdx = int(yTrue[i] / trueWidth)
            predIdx = int(yPred[i] / predWidth)

            confusionMatrix[trueIdx, predIdx] += 1

        trueStr = [f'{trueValues[i]: .{precision}f} - {trueValues[i + 1]: .{precision}f}' for i in range(numTrue - 1)]
        predStr = [f'{predValues[i]: .{precision}f} - {predValues[i + 1]: .{precision}f}' for i in range(numPred - 1)]

        confusionMatrixDf = pd.DataFrame(
            confusionMatrix,
            index=trueStr,
            columns=predStr
        )

        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(confusionMatrixDf, annot=True)
        ax.set_ylim(numTrue, 0)
        ax.set_xlim(0, numPred)

        plt.show()

    @staticmethod
    def plotConfusionMatrix(yTrue, yPred, width, precision=2):
        """

        :param yTrue:
        :param yPred:
        :param width:
        :param precision:
        :return:
        """

        assert yTrue.shape == yPred.shape and len(yTrue.shape) == 1

        minValue = min(yTrue.min(), yPred.min())
        maxValue = max(yTrue.max(), yPred.max())

        values = np.arange(minValue, maxValue + width, width)
        numValues = values.shape[0]
        confusionMatrix = np.zeros((numValues - 1, numValues - 1))

        n = yTrue.shape[0]
        for i in range(n):
            trueIdx = int(yTrue[i] / width)
            predIdx = int(yPred[i] / width)

            confusionMatrix[trueIdx, predIdx] += 1

        plotLabel = [f'{values[i]: .{precision}f} - {values[i + 1]: .{precision}f}' for i in range(numValues - 1)]

        confusionMatrixDf = pd.DataFrame(
            confusionMatrix,
            index=plotLabel,
            columns=plotLabel
        )

        plt.figure(figsize=(12, 12))
        ax = sns.heatmap(confusionMatrixDf, annot=True)
        ax.set_ylim(numValues, 0)
        ax.set_xlim(0, numValues)

        plt.show()
