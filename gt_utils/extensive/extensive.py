import itertools
import numpy as np
import nashpy as nash

class ExtensiveFormGame():
    """
    A class to represent an extensive form game.

    Attributes
    ----------
    root : Node
        The root node of the game tree.

    Methods
    -------
    get_payoffs(strategy0, strategy1):
        Computes the payoffs for the given strategies.
        
    get_normal_form():
        Converts the extensive form game to its normal form representation.
        
    enumerate_pure_strategies():
        Enumerates all pure strategies for the game.
        
    backward_induction():
        Solves the game using backward induction.
    """
    def __init__(self, root):
        self.root = root
        generate_node_identifiers(self.root)        

    def get_payoffs(self, strategy0, strategy1):
        return get_payoffs(self.root, strategy0, strategy1)
    
    def enumerate_pure_strategies(self):
        return enumerate_pure_strategies(self.root)

    def get_normal_form(self):
        "Also creates the Nashpy game"
        u1, u2 = get_normal_form(self.root)
        self.A = u1
        self.B = u2
        self.nashpy_game = nash.Game(u1, u2)
        return u1, u2

    def backward_induction(self):
        return backward_induction(self.root)
    
def generate_node_identifiers(node):
    """
    Generate identifiers for each node in the game tree.

    This function assigns an identifier to each decision node in the game tree based on the
    depth of the node and the order in which it was visited during a depth-first traversal.

    Args:
        node (Node): The starting node of the game tree.

    Notes:
        - The `node` parameter should have attributes `max_depth`, `descendants`, `is_leaf`, and `identifier`.
        - The `descendants` attribute of a node should be a list of all child nodes (not only inmediate children).
        - The `identifier` attribute of a node should be a list containing two integers.
    """
    max_depth = node.max_depth
    depth_counter = [0 for _ in range(max_depth)]
    depth_counter[0] = 1
    node.identifier = [0,0]
    for node in node.descendants:
        if node.is_leaf:
            continue
        node.set_attrs( {"identifier":[ node.depth-1, depth_counter[node.depth-1] ]} )
        depth_counter[node.depth-1] += 1

def get_payoffs(node, strategy0, strategy1):
    """
    Calculate the payoffs for a given node and strategies.

    This function traverses the game tree from the given node according to the provided strategies
    for two players and returns the payoff at the leaf node reached.

    Args:
        node (Node): The starting node of the game tree.
        strategy0 (list): The strategy for player 0, represented as a list of actions.
        strategy1 (list): The strategy for player 1, represented as a list of actions.

    Returns:
        list: The payoff at the leaf node reached by following the strategies.

    Notes:
        - The function assumes that the game tree alternates between players at each depth.
        - The `node` parameter should have attributes `max_depth`, `children`, `name`, and `is_leaf`.
        - The `children` attribute of a node should be a list of child nodes.
        - The `name` attribute of a node should be a string where the last character represents the action taken to reach that node.
        - The `is_leaf` attribute of a node should be a boolean indicating whether the node is a leaf node.
        - The `payoff` attribute of a leaf node should be a list representing the payoff for both players.
    """
    joint_strategies = [strategy0, strategy1]
    for depth in range(node.max_depth-1):
        player = 0 if depth % 2 == 0 else 1
        action = joint_strategies[player][depth//2][node.identifier[1]]
        available_actions = [ child.name[-1] for child in node.children ]
        action_index = available_actions.index(action)
        node = node.children[action_index]
        if node.is_leaf:
            return node.payoff
        
def get_normal_form(node):
    """
    Converts an extensive form game represented by the given node into its normal form.
    Parameters:
    node (Node): The root node of the extensive form game tree.
    Returns:
    tuple: A tuple containing two numpy arrays (u1, u2) representing the payoff matrices for player 1 and player 2, respectively.
    """
    pure_strategies = enumerate_pure_strategies(node)
    u1, u2 = np.zeros((len(pure_strategies[0]), len(pure_strategies[1]))), np.zeros((len(pure_strategies[0]), len(pure_strategies[1])))
                                                                        
    for i, strategy1 in enumerate(pure_strategies[0]):
        for j, strategy2 in enumerate(pure_strategies[1]):
            payoffs = get_payoffs(node, strategy1, strategy2)
            u1[i, j], u2[i, j] = payoffs
    return u1, u2   

def enumerate_pure_strategies(node):
    """
    Enumerates all pure strategies for a given game tree with root node.
    This function traverses the game tree starting from the given node and 
    generates all possible pure strategies for each player. A pure strategy 
    is a complete plan of action for a player, specifying the action to take 
    at each decision node.
    Args:
        node (Node): The root node of the game tree. The node should have 
                     attributes `max_depth` (int) indicating the maximum depth 
                     of the tree and `children` (list) containing child nodes. 
                     Each child node should have a `name` attribute where the 
                     last character represents the action taken to reach that node.
    Returns:
        list: A list of two lists, where the first list contains all pure 
              strategies for player 1 and the second list contains all pure 
              strategies for player 2. Each pure strategy is represented as 
              a list of actions.
    """

    nodes = [[node]]
    strategies = [[], []]

    for depth in range(node.max_depth-1):
        nodes_depth = []
        actions_per_child = []

        for parent in nodes[-1]: #parent=decision node
            available_actions = []
            for child in parent.children:
                nodes_depth.append(child)
                available_actions.append(child.name[-1]) #action encoded in the node name
            if len(available_actions):
                actions_per_child.append(available_actions)
        player = 0 if depth % 2 == 0 else 1
        
        if len(strategies[player]):
            aux = [strategies[player].copy()]
            for actions in actions_per_child:
                aux.append(actions)
            aux = list(itertools.product(*aux))#cartesian product
            refined_strategy = []
            for strategy in aux:
                strategy_list = strategy[0].copy()
                strategy_depth = []
                for i in range(1,len(strategy)):
                    strategy_depth.append(strategy[i])
                strategy_list.append(strategy_depth)
                refined_strategy.append(strategy_list)
            strategies[player] = refined_strategy
        else:
            aux = list(itertools.product(*actions_per_child))#cartesian product
            strategies[player] = [[list(strategy)] for strategy in aux]  

        nodes.append(nodes_depth)

    return strategies

def backward_induction(node):
    """
    Perform backward induction on a game tree node to determine the payoff of add sub-game perfect equilibria.
    This function recursively traverses the game tree from the leaves to the root,
    calculating the optimal payoff for each node based on the payoffs of its children.
    The function assumes a two-player zero-sum game where players alternate turns.
    Args:
        node (Node): The current node in the game tree. The node is expected to have
                     the following attributes:
                     - is_leaf (bool): True if the node is a leaf node, False otherwise.
                     - payoff (tuple): The payoff at the node, represented as a tuple (player1_payoff, player2_payoff).
                     - children (list): A list of child nodes.
                     - depth (int): The depth of the node in the game tree.
    Returns:
        tuple: The optimal payoff for the current node, represented as a tuple (player1_payoff, player2_payoff).
    """

    if node.is_leaf:
        return node.payoff
    
    payoffs = []
    for child in node.children:
        payoffs.append(backward_induction(child))
    
    if node.depth-1 % 2 == 0:
        # Player 1's turn, maximize their payoff
        best_payoff = max(payoffs, key=lambda x: x[0])
    else:
        # Player 2's turn, maximize their payoff
        best_payoff = max(payoffs, key=lambda x: x[1])
    
    node.payoff = best_payoff
    return best_payoff
