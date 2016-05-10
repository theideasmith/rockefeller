
def gradientDescent(x, y, theta, alpha, m, iterations):
    """
    x is for testing the hypothesis
    y is the data to match with theta
    theta is a vector containing the variables to optimize
    m is the size of the sample for computing cost, mean squared error
    """
    xTrans = x.transpose()
    for i in xrange(0, iterations):
        # Predicting with theta
        hypothesis = np.dot(x, theta)

        # Error in theta
        loss = hypothesis - y

        # Mean square error
        cost = np.sum(loss ** 2) / (2 * m)

        # avg gradient per example
        gradient = np.dot(xTrans, loss) / m
        # update
        theta = theta - alpha * gradient
    return theta
