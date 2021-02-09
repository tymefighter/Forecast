import tensorflow as tf


class SaveCallback(tf.keras.callbacks.Callback):
    """ Class to save model after each epoch """

    def __init__(self, modelSave, modelSavePath):
        """
        Initialize SaveCallback Class Members

        :param modelSave: The forecasting model itself
        :param modelSavePath: Path where to save the model
        """

        super().__init__()
        self.modelSave = modelSave
        self.modelSavePath = modelSavePath

    def on_epoch_end(self, epoch, logs=None):
        """
        Saves the model at the path provided at initialization

        :param epoch: Number of the epoch which has just ended
        :param logs: metric results for this training epoch, and for the validation
        epoch if validation is performed (tensorflow docs)
        :return: None
        """

        self.modelSave.save(self.modelSavePath)
