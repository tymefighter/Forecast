import matplotlib.pyplot as plt


class Plot:
    """Static Plotting Class"""

    @staticmethod
    def plotDataCols(X, colNames=None, title='Data Plot'):
        """
        Plots each column of X as a time series in a subplot

        :param X: Matrix with each column as a time series, it is a numpy array
        of shape (T, d)
        :param colNames: Names of each column. If it is not None, then this would
        be present as the subplot title for that column
        :param title: Title of the entire plot
        :return: None
        """

        (T, d) = X.shape
        fig, ax = plt.subplots(d)

        for dim in range(d):
            ax[dim].plot(X[:, dim])

            if colNames is not None:
                ax[dim].set_title(colNames[dim])

        plt.show()

    @staticmethod
    def plotLoss(loss, title='Loss Plot', xlabel='Iterations', ylabel='Loss'):
        """
        Plots Loss vs Iterations Curve

        :param loss: 1D List or numpy array of shape (numIters,) of loss values
        which are to be plotted
        :param title: Title of the plot
        :param xlabel: x-axis Label of the plot
        :param ylabel: y-axis Label of the plot
        :return: None
        """

        plt.plot(loss, title=title, xlabel=xlabel, ylabel=ylabel)
        plt.show()
