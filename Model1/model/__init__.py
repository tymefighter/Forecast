class Model:
    """
    Class implementing the following paper,

    Name: "Modeling Extreme Events in Time Series Prediction"
    Link: http://staff.ustc.edu.cn/~hexn/papers/kdd19-timeseries.pdf
    """

    from .initial import initializeModel

    from .train import runGruOnWindow, buildMemory, \
        trainOneTimestep, trainOneSeq, trainModel

    from .predict import computeAttentionWeights, \
        predictOneTimestep, predictOutput

    from .saveLoad import saveModel, loadModel

    def __init__(
        self,
        memorySize,
        windowSize,
        threshold,
        inputDimension,
        hiddenStateSize,
        extremeValueIndex,
        optimizer,
        extremeLossWeight,
        modelPath = None
    ):
        """Initialize the Model
    
        memorySize: Size of the memory (i.e. size of S and q),
        this is a scalar
        windowSize: Size of a window, this is a scalar
        threshold: Threshold value, above which an event is
        regarded as an extreme event
        inputDimension: Dimension of each input vector, it is a
        scalar
        hiddenStateSize: Dimension of the hidden state vector of
        the GRU, it is a scalar
        extremeValueIndex: Extreme Value Index parameter for the
        extreme value loss function, this is a scalar
        extremeLossWeight: Weight given to the extreme value loss,
        this is a scalar
        optimizer: The optimizer to be used for training the model

        modelPath: When this is not None, all the other provided
        input is ignored, the model is loaded from this path

        Initialize the model parameters and hyperparameters of the model
        """



        if modelPath is not None:
            self.loadModel(modelPath)
        else:
            self.initializeModel(
                memorySize,
                windowSize,
                threshold,
                inputDimension,
                hiddenStateSize,
                extremeValueIndex,
                extremeLossWeight,
                optimizer
            )

    def train(
        self,
        X, 
        Y, 
        seqLength,
        currTimestep = None,
        modelFilepath = None
    ):
        """Train Model on Dataset

        X: The entire input Sequence, it has shape (n, d)
        Y: The entire target Sequence, it has shape (n,)
        seqLength: Length of each sequence, imporant: each sequence
        except maybe the last would have length equal to seqLength
        currTimestep: We have to begin from here if not None and
        has value greater than or equall to windowSize, else we begin
        from windowSize
        modelFilepath: Save model parameters to this path after every
        sequence if not None, else don't save 

        Train the model using the provided data and information
        """
        self.trainModel(
            X, 
            Y, 
            seqLength,
            currTimestep,
            modelFilepath
        )

    def predict(self, X):
        """Predict Output For an Input Sequence
    
        X: The entire input Sequence, it has shape (n, d)

        Returns a sequence of outputs for the corresponding input
        sequence, it has shape (n,)
        """
        return self.predictOutput(X)

    def save(self, modelFilepath):
        """Save the Model Information

        modelFilepath: Path where to save the model information
        """
        self.saveModel(modelFilepath)

    def load(self, modelFilepath):
        """Load the Model Information

        modelFilepath: Path from where to load the model information
        """
        self.loadModel(modelFilepath)