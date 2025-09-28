import numpy as np
from typing import cast
from infoset import InfoSet, Player, Action
import os
import pickle
import random
from rlcard.utils.utils import *

import pyspiel


class CFR:
   
    def __init__(self, game):
        self.game = game
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
        if state.is_terminal():
            # Terminal state get returns.
            return state.returns()[i]
        

        elif state.is_chance_node():
            # If this is a chance node, sample an action
            # Have to do this now cause we're not using the gym environment
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = np.random.choice(chance_outcome, p=chance_proba)
            return self.walk_tree(state.child(action), i, pi_i, pi_neg_i)
        


        current_player = state.current_player()
        legal_actions = state.legal_actions()
        info_set_key = self.get_info_set_key(state, current_player)
        

        I = self._get_info_set(info_set_key=info_set_key,legal_actions=legal_actions)

        v_sigma = 0
        v_a = {}
        

        for action in I.actions():

            if current_player == i:
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
    


    def take_action(self,state):
        '''
        Helper function I'm writing so that the agent can interface with the game_wrapper class

        Args: 
            state: (pyspiel.State)
        Returns:
            some action samples from the possible states
        '''
        key = self.get_info_set_key(state, state.current_player())
        legal_actions = state.legal_actions()

        if key not in self.info_sets:
            # Assign uniform probabilities if the info set is not found
            action_probs = [1.0 / len(legal_actions) for _ in legal_actions]
        else:
            info_set = self.info_sets[key]
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
        return action


    
    #Ok, I think we did it, can come back later for some better information logging
    def train(self, iterations = 1):

        # Loop for `iterations` times
        for t in range(iterations):
            #We're only really ever doing this for 2 players
            initial_state = self.game.new_initial_state()
            for i in range(2):
                self.walk_tree(initial_state, cast(Player, i), 1, 1)     

        #     # Save checkpoints every $1,000$ iterations
        #     if (t + 1) % 1_000 == 0:
        #         experiment.save_checkpoint()

        # # Print the information sets
        # logger.inspect(self.info_sets)

    def save(self, model_path = './cfr_model'):
        ''' Save model
        '''
        with open(model_path, "wb") as f:
            pickle.dump(self.info_sets, f)

        f.close()

    def load(self, model_path = './cfr_model'):
        ''' Load model
        '''

        with open(model_path, "rb") as policy_file:
            self.info_sets = pickle.load(policy_file)
        
        policy_file.close()

       
