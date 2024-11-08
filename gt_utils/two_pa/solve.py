import numpy as np

def compute_min_max_payoff(game, accuracy = 0.001):
    """
    Compute the min-max payoff for a two-player game.
    Parameters:
    game (object): An object representing the game, which contains the payoff matrices for both players.
    accuracy (float, optional): The step size for the probability range. Default is 0.001.
    Returns:
    numpy.ndarray: A 1D array containing the min-max payoffs for both players.
    """
    #possible improvement: check borders and crossing for max accuracy
    A = game.payoff_matrices[0]
    B = game.payoff_matrices[1]

    u_minmax = np.zeros(2)
    #player 0
    p = np.arange(0, 1+accuracy, accuracy)
    u = np.zeros((2, len(p)))
    u[0,:] = A[0,0]*p+ A[0,1]*(1-p)
    u[1,:] = A[1,0]*p+ A[1,1]*(1-p)
    u_minmax[0] = np.min(np.max(u, axis=0))
    #player 1
    u = np.zeros((2, len(p)))
    u[0,:] = B[0,0]*p+ B[1,0]*(1-p)
    u[1,:] = B[0,1]*p+ B[1,1]*(1-p)
    u_minmax[1] = np.min(np.max(u, axis=0))

    return u_minmax