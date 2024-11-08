import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from gt_utils.two_pa.solve import compute_min_max_payoff

font = {'family' : 'sans-serif',
    'weight' : 'normal',
    'size'   : 20}

matplotlib.rc('font', **font)
def sort_per_polar_angle(x):
    center_of_mass = np.mean(x, axis=0)
    angles = np.arctan2(x[:, 1] - center_of_mass[1], x[:, 0] - center_of_mass[0])
    return x[np.argsort(angles)]

class UtilityPloter():
    """
    General class for plotting utility functions for 2 players 2 actions games.
    Attributes:
    -----------
    game : object
        An instance of a game containing payoff matrices for two players.
    A : numpy.ndarray
        Payoff matrix for player 1.
    B : numpy.ndarray
        Payoff matrix for player 2.
    Methods:
    --------
    make_2d_plots(player=1):
        Generates 2D plots of the utility functions for the specified player.
        Parameters:
        -----------
        player : int, optional
            The player for whom the utility function is plotted (1 or 2). Default is 1.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.AxesSubplot
            The axes object containing the plot.
    make_3d_plots(player=1):
        Generates 3D plots of the utility functions for the specified player.
        Parameters:
        -----------
        player : int, optional
            The player for whom the utility function is plotted (1 or 2). Default is 1.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes._subplots.Axes3DSubplot
            The axes object containing the plot.
    Notation:
    ---------
    The payoff matrices A and B are represented as follows:
    A = | a11  a12 |
        | a21  a22 |
    B = | b11  b12 |
        | b21  b22 |
    where aij and bij are the payoffs for player 1 and player 2 respectively, given their actions.
    """
    def __init__(self, game) -> None:
        self.game = game
        self.A = game.payoff_matrices[0]
        self.B = game.payoff_matrices[1]
    def make_2d_plots(self, player=1):
        """
        Notation: 
        - u_1(A_i,s_2)$: utility of player 1 when playing action A_i and the other player plays strategy s_2.
        s_2 consists of playing A_1 or A_2 with probability p (s_2(A_1)) or 1-p (s_2(A_2)) respectively.
        - $u_2(s_1, A_i)$: utility of player 2 when playing action A_i and the other player plays strategy s_1.
        s_1 consists of playing A_1 or A_2 with probability p (s_1(A_1)) or 1-p (s_1(A_2)) respectively.
        """
        fig, ax = plt.subplots()
        if player == 1:
            p = np.arange(0, 1, 0.01)
            u1 = self.A[0,0]*p+ self.A[0,1]*(1-p)
            u2 = self.A[1,0]*p+ self.A[1,1]*(1-p)
            ax.plot(p, u1, label='$u_1(A_1,s_2)$')
            ax.plot(p, u2, label='$u_1(A_2,s_2)$')
            ax.set_xlabel('$s_2(A_1)$')
            ax.set_ylabel('$u_1$')
        if player == 2:
            p = np.arange(0, 1, 0.01)
            u1 = self.B[0,0]*p+ self.B[1,0]*(1-p)
            u2 = self.B[0,1]*p+ self.B[1,1]*(1-p)
            ax.plot(p, u1, label='$u_2(s_1, A_1)$')
            ax.plot(p, u2, label='$u_2(s_1, A_2)$')
            ax.set_xlabel('$s_1(A_1)$')
            ax.set_ylabel('$u_2$')
        ax.legend()
        plt.tight_layout()
        return fig, ax
    def make_3d_plots(self, player=1):
        """
        Notation:
        - $u_1$: utility of player 1.
        - $u_2$: utility of player 2.
        - $s_1(A_1)$: probability of playing action A_1 by player 1.
        - $s_2(A_1)$: probability of playing action A_1 by player 2.
        """
        # Create a meshgrid for the probabilities
        p1 = np.linspace(0, 1, 100)
        p2 = np.linspace(0, 1, 100)
        P1, P2 = np.meshgrid(p1, p2)
        if player == 1:
            # Calculate the utility for player 1
            U1 = self.A[0, 0] * P1 * P2 + self.A[0, 1] * P1 * (1 - P2) + self.A[1, 0] * (1 - P1) * P2 + self.A[1, 1] * (1 - P1) * (1 - P2)

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(P1, P2, U1, cmap='viridis')

            ax.set_xlabel('$s_1(A_1)$', labelpad=16)
            ax.set_ylabel('$s_2(A_1)$', labelpad=16)
            ax.set_zlabel('$u_1$')
        if player == 2:
            # Calculate the utility for player 2
            U2 = self.B[0, 0] * P1 * P2 + self.B[0, 1] * P1 * (1 - P2) + self.B[1, 0] * (1 - P1) * P2 + self.B[1, 1] * (1 - P1) * (1 - P2)

            # Plotting
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(P1, P2, U2, cmap='viridis')

            ax.set_xlabel('$s_1(A_1)$', labelpad=16)
            ax.set_ylabel('$s_2(A_1)$', labelpad=16)
            ax.set_zlabel('$u_2$')
        plt.tight_layout()
        return fig, ax
    
class FolkPlotter():
    "Class for plotting regions of possible payoffs for NE of infinitely-repeated 2 players 2 actions games"
    def __init__(self, game) -> None:
        self.game = game
        self.A = game.payoff_matrices[0]
        self.B = game.payoff_matrices[1]
    def make_folk_plot(self):
        min_max = compute_min_max_payoff(self.game)

        vertex = np.array([[self.A[0,0], self.B[0,0]], [self.A[0,1], self.B[0,1]], [self.A[1,0], self.B[1,0]], [self.A[1,1], self.B[1,1]]])

        vertex = sort_per_polar_angle(vertex)
        vertex = np.vstack([vertex, vertex[0]])

        fig, ax = plt.subplots()

        ax.plot(vertex[:, 0], vertex[:, 1], '--', label='Feasible')
        y = np.arange(np.min(vertex[:, 1]), np.max(vertex[:, 1]), 0.01)
        x = [min_max[0] for _ in range(len(y))]
        ax.plot(x, y, '--', label='Enfoceable', color="orange")
        x = np.arange(np.min(vertex[:, 0]), np.max(vertex[:, 0]), 0.01)
        y = [min_max[1] for _ in range(len(x))]
        ax.plot(x, y, '--', color="orange")
        ax.legend()

        ax.set_xlabel('$u_1$')
        ax.set_ylabel('$u_2$')
        plt.tight_layout()
        return fig, ax