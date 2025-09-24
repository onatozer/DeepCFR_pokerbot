import numpy as np
from typing import cast
from infoset import InfoSet, Player, Action
import os
import pickle
import random
from rlcard.utils.utils import *

import pyspiel


class CFR:
   
    def __init__(self):
        self.info_sets = {}

    def get_info_set_key(self, state: pyspiel.State, current_player):
        """
        Generates a unique key for the information set based on the player's observations.

        Args:
            player_id (int): The ID of the player for whom the key is being generated.

        Returns:
            bytes: A byte representation of the player's observation, used as a key for the info set.
        """
        state_str = state.information_state_string(current_player)
        return state_str

   
    def _get_info_set(self, info_set_key, legal_actions):
        """
        Returns the information set I for the current player at a given state.
        """
        if info_set_key not in self.info_sets:
            self.info_sets[info_set_key] = InfoSet(info_set_key, legal_actions)
        return self.info_sets[info_set_key]


    def walk_tree(self, state: pyspiel.State, i: Player, pi_i: float, pi_neg_i: float) -> float:
        #player 1 ->index 0, player 2 -> index 1
        if self.env.is_terminal():
            return self.env.returns()[i]
        

        elif state.is_chance_node():
            # If this is a chance node, sample an action
            # Have to do this now cause we're not using the gym environment
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = np.random.choice(chance_outcome, p=chance_proba)
            return self.traverse(state.child(action), i, pi_i, pi_neg_i)
        


        current_player = state.current_player()
        legal_actions = state.legal_actions()
        info_set_key = self.get_info_set_key(state, current_player)
        

        I = self._get_info_set(info_set_key=info_set_key,legal_actions=legal_actions)

        v_sigma = 0
        v_a = {}


        for action in I.actions():

            if self.env.get_player_id() == i:
                v_a[action] = self.walk_tree(state.child(action), i, pi_i * I.strategy[action], pi_neg_i)

            else:
                v_a[action] = self.walk_tree(state.child(action), i, pi_i, pi_neg_i  * I.strategy[action])

            v_sigma += I.strategy[action] * v_a[action]


        if current_player == i:
            for action in I.actions():
                I.regret[action] += pi_neg_i*(v_a[action] - v_sigma)
                I.cumulative_strategy[action] += pi_i*I.strategy[action]

            
        I.calculate_strategy()
        
        return v_sigma

    
    def eval_step(self, state):
        edited_state = self._state_abstraction(state)
        legal_actions = list(state['legal_actions'].keys())

        if edited_state not in self.info_sets:
            # Assign uniform probabilities if the info set is not found
            action_probs = [1.0 / len(legal_actions) for _ in legal_actions]
        else:
            info_set = self.info_sets[edited_state]
            average_strategy = info_set.get_average_strategy()
            # Get probabilities for legal actions
            action_probs = [average_strategy.get(a, 0.0) for a in legal_actions]

        # Normalize action_probs to sum to 1
        sum_probs = sum(action_probs)
        if sum_probs > 0:
            action_probs = [p / sum_probs for p in action_probs]
        else:
            # If the sum is zero (all probabilities are zero), assign uniform probabilities
            action_probs = [1.0 / len(legal_actions) for _ in legal_actions]

        # Now, action_probs should sum to 1
        action = np.random.choice(legal_actions, p=action_probs)
        return action, action_probs


    
    #Ok, I think we did it, can come back later for some better information logging
    def train(self, epochs = 1):
        """
        ### Iteratively update $\textcolor{lightgreen}{\sigma^t(I)(a)}$

        This updates the strategies for $T$ iterations.
        """

        # Loop for `epochs` times
        for t in range(epochs):
            for i in range(self.n_players):
                self.env.reset()
                self.walk_tree(cast(Player, i), 1, 1)     

        #     # Save checkpoints every $1,000$ iterations
        #     if (t + 1) % 1_000 == 0:
        #         experiment.save_checkpoint()

        # # Print the information sets
        # logger.inspect(self.info_sets)

    def save(self, model_path = './cfr_model'):
        ''' Save model
        '''
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        cfr_policy = open(os.path.join(model_path, 'policy.pkl'),'wb')
        pickle.dump(self.info_sets, cfr_policy)
        cfr_policy.close()

    def load(self, model_path = './cfr_model'):
        ''' Load model
        '''
        if not os.path.exists(model_path):
            print(f'No model found at {model_path}')
            return

        policy_file = open(os.path.join(model_path, 'policy.pkl'),'rb')
        self.info_sets = pickle.load(policy_file)
        policy_file.close()

       
