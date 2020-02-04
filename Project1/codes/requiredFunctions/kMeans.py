import numpy as np
from time import time


def kMeansBatch(data, K, rand_init=True, init_values=(0,0), return_cost=False, iter_ind=False):
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
    start = time()
    num_points = data.shape[0]
    if rand_init:
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
    old_means = []

    while not np.array_equal(means,prev_means) and total_iterations < 500:
        total_iterations+=1
        ind_vars = np.zeros((num_points,K))
        distances = np.zeros((num_points,K))
        # calculate the distance to each mean from each point
        for j in range(K):
            # reinitialize mu_j before use if there are any NaNs
            if np.isnan(means[j]).any():
                means[j] = data[np.random.randint(0,num_points)]
            distances[:, j] = np.array([pow(np.linalg.norm(point - means[j]),2)
                                        for point in data]).T

        # determine which mean is closest
        I = np.argmin(distances, axis=1)
        # create a OHE array, with rows representing points and columns
        # representing means, showing to which mean each point belongs
        ind_vars[range(num_points), I[range(num_points)]] = 1
        prev_means = means.copy()
        old_means.append(prev_means)
        for j in range(K):
            if ind_vars[:,j].sum() == 0:
                means[j] = prev_means[j]
            else:
                means[j] = (data.T*ind_vars[:,j]).sum(axis=1)/ind_vars[:,j].sum()

        if iter_ind:
            elpsd_time = round((time() - start)/60.0,2)
            print('iterations: ',total_iterations, 'elapsed time (min): ', elpsd_time, end='\r')
        
    old_means = np.array(old_means)

    if return_cost:
        cost = []
        for j in range(K):
            for i in range(num_points):
                if I[i] == j:
                    cost.append(pow(np.linalg.norm(data[i]-means[j]),2))

        cost = np.array(cost).sum()

        return means, ind_vars, total_iterations, old_means, cost
    else:
        return means, ind_vars, total_iterations, old_means

def kMeansOnline(data, K, eta, thresh, rand_init=True, init_values=(0,0)):
    """
    Implementation of the on-line K-means clustering algorithm.

    Arguments:
               data - Array-like data set to be clustered.
                  K - Integer number of centroids to use for clustering.
                eta - Float-type learning rate, or the fraction of the distance
                      to move the means toward the closest points.
             thresh - Float-type value identifying the minimum change in means
                      for which they shall be considered effectively the same,
                      and thus convergence has been achieved.
        random_init - Boolean expressing desire to randomly initialize the
                      mean values to existing data points. If False, uses
                      values provided in init_values arg.
        init_values - K-length tuple or array-like values to be used to
                      initialize the K means. If tuple (x1,x2,...,xk), will
                      initialize each mean to ((x1,x1,...), (x2,x2,...), ...,
                      (xk,xk,...)).
    """
    num_points = data.shape[0]
    if rand_init:
        means = data[np.random.randint(0,num_points,K)]
    else:
        if type(init_values)==np.ndarray:
            means = init_values
        else:
            if len(init_values)<K:
                init_values = init_values+tuple(np.random.rand(10,
                    K-len(init_values)))
            means = (np.ones((K,data.shape[1])).T*init_values).T
    prev_means = means+1  # +100 just to make it different for first iteration
    total_epochs = 0
    below_thresh = False
    old_means = []

    while not below_thresh and total_epochs < 1000:
        np.random.shuffle(data)
        total_epochs+=1
        ind_vars = np.zeros((num_points,K))
        distances = np.zeros((num_points,K))
        # calculate the distance to each mean from each point
        for j in range(K):
            # reinitialize mu_j before use if there are any NaNs
            if np.isnan(means[j]).any()\
               or np.isclose(means[0],means[1],atol=1e-02).any():
                means[j] = data[np.random.randint(0,num_points)]
            distances[:, j] = np.array([np.linalg.norm(point - means[j])**2
                                        for point in data]).T

        # determine which mean is closest
        I = np.argmin(distances, axis=1)
        # create a OHE array, with rows representing points and columns
        # representing means, showing to which mean each point belongs
        ind_vars[range(num_points), I[range(num_points)]] = 1
        prev_means = means.copy()
        old_means.append(prev_means)
        # calculate new means
        for n in range(num_points):
            for i in range(K):
                if I[n]==i:
                    means[i] = prev_means[i] + eta*(data[n]-prev_means[i])
        
        # determine if the threshold condition has been met now
        epoch_diffs = []
        for k in range(K):
            epoch_diffs.append(np.linalg.norm(means[k]-prev_means[k]))
        epoch_diff = np.mean(epoch_diffs)
        print(epoch_diff, end='\r')
        if epoch_diff<thresh:
            below_thresh = True
        
    old_means = np.array(old_means)

    return means, ind_vars, total_epochs, old_means

