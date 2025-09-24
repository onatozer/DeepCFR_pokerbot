import sys

from typing import List, cast, Dict, NewType
import numpy as np
import torch
from config import *





'''
Args:
    observation (dict): The observation of the current state.
    {
        "legal_actions": List of the Actions that are legal to take.    N
        "street": 0, 1, or 2 representing pre-flop, flop, or river respectively    Y
        "my_cards": List[str] of your cards, e.g. ["1s", "2h"]      Y
        "board_cards": List[str] of the cards on the board         Y

        "my_pip": int, the number of chips you have contributed to the pot this round of betting            Y
        "opp_pip": int, the number of chips your opponent has contributed to the pot this round of betting      Y

        "my_stack": int, the number of chips you have remaining     Y
        "opp_stack": int, the number of chips your opponent has remaining       Y (might not be needed as information is already encoded in mystack - 2p zero sum)

        "my_bankroll": int, the number of chips you have won or lost from the beginning of the game to the start of this round      N

        "min_raise": int, the smallest number of chips for a legal bet/raise        N
        "max_raise": int, the largest number of chips for a legal bet/raise         N
    }

    info set key format:
    str(observation["street"]) + ''.join(observation["my_cards"]) + ''.join(observation["board_cards"]) + str(observation["my_stack"]) + str(observation["opp_stack"]) 

    Inputs for nn:
    
'''


class InfoSet:
    """
    Information Set I_i
    """

    def __init__(self, key, legal_actions: List[int]):
        """
        Initialize the InfoSet with a key and the legal actions available at this state.
        """
        self.key = key
        self.legal_actions = legal_actions

        self.regret = {a: 0.0 for a in self.legal_actions}
        self.cumulative_strategy = {a: 0.0 for a in self.legal_actions}
        self.strategy = {}
        self.calculate_strategy()


    #TODO: Will need to change for poker env with continous raise amount
    def actions(self) -> List[int]:
        """
        Return the list of legal actions.
        """
        return self.legal_actions

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
                self.strategy[action] = max(regret, 0) / sum_regrets

    def get_average_strategy(self):
        """
        Get average strategy based on cumulative strategy.
        """
        strategy_sum = sum(self.cumulative_strategy.values())

        avg_strategy = {a: self.cumulative_strategy.get(
            a, 0.) for a in self.actions()}

        for action in self.legal_actions:
            avg_strategy[action] = self.cumulative_strategy[action]/strategy_sum

        return avg_strategy


    #TODO: This also needs to change for openSpiel (I basically changed nothing, but should work fine)
    def convert_key_to_tensor(self):
        '''
        key:
            hole_cards + board_cards + bet_features
        '''

        # One numpy array
        key = self.key

        key = np.frombuffer(key, dtype=np.int32)


        hole_cards = key[:2]


        board_cards = key[2:7]
        
        card_tensor = [
            torch.tensor(np.array(hole_cards).reshape(1, -1)).to(DEVICE), #preflop
            torch.tensor(np.array(board_cards[0:3]).reshape(1, -1)).to(DEVICE), #flop
            torch.tensor(np.array(board_cards[3:4]).reshape(1, -1)).to(DEVICE), #turn
            torch.tensor(np.array(board_cards[4:5]).reshape(1, -1)).to(DEVICE), #river
        ]

        """
        cards = [
            # Hole cards for datapoint 1 and datapoint 2 (2 cards each) card round 1
            torch.tensor([[3, 12]]),
            # Board 1 cards for datapoint 1 and datapoint 2 (1 card each) card round 2
            torch.tensor([[10]]),
            # Board 2 cards for datapoint 1 and datapoint 2 (1 card each) card round 3, -1 if card round not reached
            torch.tensor([[11]])
        ]"""

        # print(f"key {key}")
        bet_features = key[7:]
        bet_tensor = torch.tensor(bet_features.reshape(1, -1)).float()

        # print(f"returning card tensor {card_tensor}\nand returning bet tensor {bet_tensor}")
        return card_tensor, bet_tensor.to(DEVICE)

