def computeAttentionWeights(
    self
    state,
):
    return tf.squeeze(tf.nn.softmax(np.dot(
        self.S,
        state
    )))