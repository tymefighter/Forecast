import tensorflow as tf

EPS = 1e-8

def squareLoss(yPred, yTrue):
    """Square Loss Between Prediction and Target
    
    yPred: predicted values, it has shape (n,)
    yTrue: target values, it a also has shape (n,)

    Returns the square loss between predicted
    and target values, is is a scalar
    """
    return tf.math.reduce_sum(tf.math.square(yPred - yTrue))

def extremeValueLoss(
    pred, 
    target,
    numNormalEvents,
    numExtremeEvents,
    extremeValueIndex
):
    """Extreme Value Loss

    pred: Prediction that extreme event occurred, shape: (n,)
    target: Target which tells whether extreme event occurred
    or not, shape: (n,)
    numNormalEvents: Number of normal events seen uptil now,
    it is a scalar
    numExtremeEvents: Number of extreme events seen uptil now,
    it is a scalar
    extremeValueIndex: The extreme value index parameter,
    it is a scalar

    Returns the extreme value loss, it is a scalar
    """

    totalEvents = numNormalEvents + numExtremeEvents
    fractionNormal = numNormalEvents / totalEvents
    fractionExtreme = numExtremeEvents / totalEvents

    extremePart = - fractionNormal \
        * tf.math.pow(
            1 - pred / extremeValueIndex, 
            extremeValueIndex
        ) \
        * target \
        * tf.math.log(pred + EPS)

    normalPart = - fractionExtreme \
        * tf.math.pow(
            1 - (1 - pred) / extremeValueIndex, 
            extremeValueIndex
        ) \
        * (1 - target) \
        * tf.math.log(1 - pred + EPS)

    return tf.math.reduce_sum(extremePart + normalPart)

def loss1(
    yPred, 
    yTrue, 
    extremePred, 
    extremeTarget,
    extremeWeight,
    numNormalEvents,
    numExtremeEvents,
    extremeValueIndex
):
    """ Loss Function For the Model
    
    yPred: Final Predicted Output, shape: (n,)
    yTrue: Actual Target, shape: (n,)
    extremePred: Prediction that extreme event would occur, shape: (n,)
    extremeTarget: Target that extreme event occured, shape: (n,)
    extremeWeight: Weight given to the extreme part of the loss (i.e.
    the extreme value loss), it is a scalar
    numNormalEvents: Number of normal events seen uptil now, it is
    a scalar
    numExtremeEvents: Number of extreme events seen uptil now, it is
    a scalar
    extremeValueIndex: The extreme value index parameter,
    it is a scalar

    Returns the Loss value for the given input, the returned value is
    a scalar
    """
    return squareLoss(yPred, yTrue) + \
            extremeWeight * extremeValueLoss(
                extremePred, 
                extremeTarget,
                numNormalEvents,
                numExtremeEvents,
                extremeValueIndex
            )

"""Loss Function for Enchancing Memory

This is just extremeValueLoss, but is aliased for better readability,
look up the doc for extremeValueLoss to get more information about it
"""
loss2 = extremeValueLoss