import numpy as np

def generate_uniform_train_test_set(dof, train_set_size, train_q_lims, train_p_lims, test_set_size, test_q_lims, test_p_lims, rng):
    """
    Uniformly places the train and test points within the given domain. Test set is sampled distinctly.

    @param dof              : degree of freedom of the system == decides dimensions of the sampled points
    @param train_set_size   : number of total points to sample for train set
    @param train_q_lims     : domain for q (position) dimensions, as list([min,max])
    @param train_p_lims     : domain for p (momentum) dimensions, as list([min,max])
    @param ... defined analogously for test
    @param rng              : random number generator

    @return x_train, x_test stacked up as (n_points, 2*dof) dimensional arrays
    """
    q_train_grid = []
    p_train_grid = []

    for d in range(dof):
        # sample points randomly
        q_train_grid.append(rng.uniform(low=train_q_lims[d][0], high=train_q_lims[d][1], size=(train_set_size)))
        p_train_grid.append(rng.uniform(low=train_p_lims[d][0], high=train_p_lims[d][1], size=(train_set_size)))

    q_train_grid = np.array(q_train_grid)
    p_train_grid = np.array(p_train_grid)

    # sample distinct test samples
    q_test_grid = []
    p_test_grid = []

    for d in range(dof):
        q_test_grid.append([])
        while len(q_test_grid[d]) < test_set_size:
            candidate_samples = rng.uniform(low=test_q_lims[d][0], high=test_q_lims[d][1], size=(test_set_size - len(q_test_grid[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, q_train_grid[d], assume_unique=True)
            q_test_grid[d].extend(new_samples)

        p_test_grid.append([])
        while len(p_test_grid[d]) < test_set_size:
            candidate_samples = rng.uniform(low=test_p_lims[d][0], high=test_p_lims[d][1], size=(test_set_size - len(p_test_grid[d])))
            # setdiff1d returns all elements in arr1 that are not in arr2
            new_samples = np.setdiff1d(candidate_samples, p_train_grid[d], assume_unique=True)
            p_test_grid[d].extend(new_samples)

    q_test_grid = np.array(q_test_grid)
    p_test_grid = np.array(p_test_grid)

    # column stacked (q_i, p_i): (N, 2*dof)
    x_train = np.column_stack([ q.flatten() for q in q_train_grid ] + [ p.flatten() for p in p_train_grid ])
    x_test = np.column_stack([ q.flatten() for q in q_test_grid ] + [ p.flatten() for p in p_test_grid ])

    assert x_train.ndim == 2
    assert x_train.shape[1] == 2 * dof
    assert x_test.ndim == 2
    assert x_test.shape[1] == 2 * dof

    return x_train, x_test

def generate_grid(N_qs, N_ps, q_lims, p_lims, dof, linspace=False, rng=None):
    """
    Generates meshgrid for Hamiltonian systems with positions q and momenta p

    This function is intended for plotting since it creates the grid linearly spaced.

    @param N_qs: number of grid points in q in each dimension as a list
    @param N_ps: number of grid points in p in each dimension as a list
    @param q_lims: list of [q_min, q_max] in each dimension
    @param p_lims: list of [p_min, p_max] in each dimension
    @param dof: degree of freedom of the system
    @param linspace: if True, the grid is linearly spaced, otherwise uniformly distributed
    @param random_seed: used in case of uniform sampling if given

    @return q_ranges, p_ranges, q_grids, p_grids
    """
    params = [N_qs, N_ps, q_lims, p_lims]
    for param in params:
        assert len(param) == dof

    q_ranges = []
    p_ranges = []

    if linspace:
        for d in range(dof):
            q_ranges.append(np.linspace(q_lims[d][0], q_lims[d][1], N_qs[d]))
            p_ranges.append(np.linspace(p_lims[d][0], p_lims[d][1], N_ps[d]))
    else:
        assert rng != None
        for d in range(dof):
            q_ranges.append(rng.uniform(low=q_lims[d][0], high=q_lims[d][1], size=(N_qs[d])))
            p_ranges.append(rng.uniform(low=p_lims[d][0], high=p_lims[d][1], size=(N_ps[d])))

    grids = np.meshgrid(*(q_ranges + p_ranges))

    # returns q_grids, p_grids
    return q_ranges, p_ranges, grids[:dof], grids[dof:]
