import tensorflow as tf

def squareLoss(yPred, yTrue):
    return tf.math.square(yPred - yTrue)

def extremeValueLoss(
    pred, 
    target,
    numNormalEvents,
    numExtremeEvents
):
    extremePart = - numNormalEvents \
        * tf.math.pow(
            1 - pred / self.extremeValueIndex, 
            self.extremeValueIndex
        ) \
        * target \
        * tf.math.log(pred)

    normalPart = - numNormalEvents \
        * tf.math.pow(
            1 - (1 - pred) / self.extremeValueIndex, 
            self.extremeValueIndex
        ) \
        * (1 - target) \
        * tf.math.log(1 - pred)

    return extremePart + normalPart

def loss1(
    yPred, 
    yTrue, 
    extremePred, 
    extremeTarget,
    extremeWeight,
    numNormalEvents,
    numExtremeEvents
):
    return squareLoss(yPred, yTrue) + \
            extremeWeight * \
            extremeValueLoss(
                extremePred, 
                extremeTarget,
                numNormalEvents,
                numExtremeEvents
            )

def loss2(
    preds, 
    targets,
    numNormalEvents,
    numExtremeEvents
):
    tf.reduce_sum(extremeValueLoss(
        preds,
        targets
    ), axis = 0)