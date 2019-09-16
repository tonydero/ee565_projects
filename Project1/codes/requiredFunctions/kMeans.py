import numpy as np


def kMeansBatch(data, K, random_init=True, init_values=(0,0)):
    """
    Implementation of the batch K-means clustering algorithm.

    Arguments:
               data - Array-like data set to be clustered.
                  K - Integer number of centroids to use for clustering.
        random_init - Boolean expressing desire to randomly initialize the
                      mean values to existing data points. If False, uses
                      values provided in init_values arg.
        init_values - K-length tuple or array-like values to be used to
                      initialize the K means. If tuple (x1,x2,...,xk), will
                      initialize each mean to ((x1,x1,...), (x2,x2,...), ...,
                      (xk,xk,...)).
    """
    num_points = data.shape[0]
    if random_init:
        means = data[np.random.randint(0,num_points,K)]
    else:
        if type(init_values)==np.ndarray:
            means = init_values
        else:
            if len(init_values)<K:
                init_values = init_values+tuple(np.random.rand(10,
                    K-len(init_values)))
            means = (np.ones((K,data.shape[1])).T*init_values).T
    prev_means = means+1  # +1 just to make it different for first iteration
    total_iterations = 0

    while not np.array_equal(means,prev_means) and total_iterations < 500:
        total_iterations+=1
        ind_vars = np.zeros((num_points,K))
        distances = np.zeros((num_points,K))
        # calculate the distance to each mean from each point
        for j in range(K):
            # reinitialize mu_j before use if there are any NaNs
            if np.isnan(means[j]).any():
                means[j] = data[np.random.randint(0,num_points)]
            distances[:, j] = np.array([np.linalg.norm(point - means[j])**2
                                        for point in data]).T

        # determine which mean is closest
        min_dist_idx = np.argmin(distances, axis=1)
        # create a OHE array, with rows representing points and columns
        # representing means, showing to which mean each point belongs
        ind_vars[range(num_points), min_dist_idx[range(num_points)]] = 1
        prev_means = means.copy()
        for j in range(K):
            means[j] = (data.T*ind_vars[:,j]).sum(axis=1)/ind_vars[:,j].sum()
        
    return means, ind_vars, total_iterations

def kMeansOnline(data, K, eta, thresh, rand_init=True, init_values=(0,0)):
    """
    Implementation of the on-line K-means clustering algorithm.

    Arguments:
               data - Array-like data set to be clustered.
                  K - Integer number of centroids to use for clustering.
                eta - Float-type learning rate.
             thresh - Float-type value identifying the minimum change in 
                      shall be considered the same.
        random_init - Boolean expressing desire to randomly initialize the
                      mean values to existing data points. If False, uses
                      values provided in init_values arg.
        init_values - K-length tuple or array-like values to be used to
                      initialize the K means. If tuple (x1,x2,...,xk), will
                      initialize each mean to ((x1,x1,...), (x2,x2,...), ...,
                      (xk,xk,...)).
    """
    num_points = data.shape[0]
    if random_init:
        means = data[np.random.randint(0,num_points,K)]
    else:
        if type(init_values)==np.ndarray:
            means = init_values
        else:
            if len(init_values)<K:
                init_values = init_values+tuple(np.random.rand(10,
                    K-len(init_values)))
            means = (np.ones((K,data.shape[1])).T*init_values).T
    prev_means = means+1  # +1 just to make it different for first iteration
    total_epochs = 0
    below_thresh = False

    while not below_thresh and total_epochs < 500:
        total_epochs+=1
        ind_vars = np.zeros((num_points,K))
        distances = np.zeros((num_points,K))
        # calculate the distance to each mean from each point
        for j in range(K):
            # reinitialize mu_j before use if there are any NaNs
            if np.isnan(means[j]).any():
                means[j] = data[np.random.randint(0,num_points)]
            distances[:, j] = np.array([np.linalg.norm(point - means[j])**2
                                        for point in data]).T

        # determine which mean is closest
        min_dist_idx = np.argmin(distances, axis=1)
        # create a OHE array, with rows representing points and columns
        # representing means, showing to which mean each point belongs
        ind_vars[range(num_points), min_dist_idx[range(num_points)]] = 1
        prev_means = means.copy()
        # calculate new means
        for j in range(K):
            means[j] = (data.T*ind_vars[:,j]).sum(axis=1)/ind_vars[:,j].sum()

        data = np.random.shuffle(data)
        
        # determine if the threshold condition has been met now
        blah = 
        if (blah<thresh).all():
            below_thresh = True
        
    return means, ind_vars, total_epochs

