import matplotlib.pyplot as plt
import numpy as np


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

        :param pred: Predictions
        :param target: True Targets
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
