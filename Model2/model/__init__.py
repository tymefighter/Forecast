class Model:
    """
    """

    def __init__(
        self,
        memorySize,
        windowSize,
        inputDimension,
        encoderStateSize,
        lstmStateSize,
        optimizer,
        extremeLossWeight,
        modelPath = None
    ):
        """Initialize the Model
        """

        if modelPath is not None:
            self.loadModel(modelPath)
        else:
            self.initializeModel(
                memorySize,
                windowSize,
                inputDimension,
                encoderStateSize,
                lstmStateSize,
                optimizer
            )

    def train(
        self,
        X, 
        Y, 
        seqLength,
        currTimestep = None,
        modelFilepath = None,
        verbose = 1
    ):
        """Train Model on Dataset

        X: The entire input Sequence, it has shape (n, d)
        Y: The entire target Sequence, it has shape (n,)
        seqLength: Length of each sequence, important: each sequence
        except maybe the last would have length equal to seqLength
        currTimestep: We have to begin from here if not None and
        has value greater than or equall to windowSize, else we begin
        from windowSize
        modelFilepath: Save model parameters to this path after every
        sequence if not None, else don't save
        verbose: 0 - no info, 1 - some info, > 1 - more info

        Train the model using the provided data and information
        """
        self.trainModel(
            X, 
            Y, 
            seqLength,
            currTimestep,
            modelFilepath,
            verbose
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