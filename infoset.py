from typing import List, cast, Dict, NewType
import numpy as np


Player = NewType('Player', int)
Action = NewType('Action', int)

class InfoSet:
    """
    Information Set I_i
    """

    def __init__(self, key: bytes, legal_actions: List[Action]):
        """
        Initialize the InfoSet with a key and the legal actions available at this state.
        """
        self.key = key
        self.legal_actions = legal_actions
        self.regret = {a: 0.0 for a in legal_actions}
        self.cumulative_strategy = {a: 0.0 for a in legal_actions}
        self.strategy = {}
        self.calculate_strategy()

    def actions(self) -> List[Action]:
        """
        Return the list of legal actions.
        """
        return self.legal_actions

    #TODO: Implement this function
    def calculate_strategy(self):
        """
        Calculate current strategy using regret matching.
        """
        # Find the sum of all the cumulative regrets at this infoset
        sum_regrets = 0
        for regret in self.regret.values():
            sum_regrets += max(regret, 0)

        if sum_regrets <= 0:
            for action in self.regret.keys():
                self.strategy[action] = 1 / len(self.regret)

        else:
            for action, regret in self.regret.items():
                self.strategy[action] = max(regret, 0) /sum_regrets
        

    #TODO: Implement this function
    def get_average_strategy(self):
        """
        Get average strategy based on cumulative strategy.
        """
        strategy_sum = sum(self.cumulative_strategy.values())

        avg_strategy = {a: self.cumulative_strategy.get(a,0.) for a in self.actions()}

        for action in self.legal_actions:
            avg_strategy[action] = self.cumulative_strategy[action]/strategy_sum

        return avg_strategy



        

    def to_dict(self):
        """
        Save the information set to a dictionary.
        """
        return {
            'key': self.key,
            'regret': self.regret,
            'average_strategy': self.cumulative_strategy,
        }

    def load_dict(self, data: Dict[tuple, any]):
        """
        Load data from a saved dictionary.
        """
        self.regret = data['regret']
        self.cumulative_strategy = data['average_strategy']
        self.calculate_strategy()

    def __repr__(self):
        """
        Human-readable string representation of the InfoSet.
        """
        total = sum(self.cumulative_strategy.values())
        if total > 0:
            strategy = {a: self.cumulative_strategy[a] / total for a in self.legal_actions}
        else:
            count = len(self.legal_actions)
            strategy = {a: 1.0 / count for a in self.legal_actions}
        return f'InfoSet(key={self.key}, strategy={strategy})'

