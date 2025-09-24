import sys
import numpy as np
from typing import cast, List, Tuple
from infoset import InfoSet
import os
import re
import pyspiel
from open_spiel.python import policy
from pyspiel import exploitability
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from config import *
from network import DeepCFRModel
from reservoir import MemoryReservoir, custom_collate
from torch.cuda.amp import GradScaler, autocast




class CFR(policy.Policy):

    def __init__(self, *,
                 n_players: int = 2,
                 game
                 ):
        all_players = list(range(game.num_players()))
        super(CFR, self).__init__(game, all_players)
        self.n_players = n_players       
        self.game = game

  

    def get_info_set_key(self, state: pyspiel.State, current_player: int):
        """
        Generates a unique key for the information set based on the player's observations.

        Args:
            player_id (int): The ID of the player for whom the key is being generated.

        Returns:

            bytes: A byte representation of the player's observation, used as a key for the info set.
            (for deepCFR, we want this representation to be an array of the following structure:
                [hole cards + board cards + bet features]
                where each card is represented as an integer
                )
        """
        state_str = state.information_state_string(current_player)


        pattern = r"\[(\w+):?\s*([^\]]*)\]"

        # Use re.findall to extract all matches
        matches = re.findall(pattern, state_str)


        state_info = {key: value for key, value in matches}

        #Initially output as single string, but we want list of individual cards, each 2 chars long, so we have to do this trickery:
        hole_cards_str = state_info["Private"]
        hole_cards_str = [hole_cards_str[i] + hole_cards_str[i + 1] for i in range(0, len(hole_cards_str) - 1, 2)]

        board_cards_str = state_info["Public"]
        board_cards_str = [board_cards_str[i]+ board_cards_str[i + 1] for i in range(0, len(board_cards_str) - 1, 2)]

        #In openSpiel, the suits are lowercase, and the ranks upper
        suit_to_int = {'s': 0, 'h': 13, 'd': 26, 'c': 39}
        rank_to_int = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8,'J': 9 ,'Q': 10, 'K': 11, 'A': 12}

        hole_cards = [-1]*NUM_HOLE_CARDS

        for i, card in enumerate(hole_cards_str):
            hole_cards[i] = rank_to_int[card[0]] + suit_to_int[card[1]]

        board_cards = [-1]*NUM_BOARD_CARDS

        for i, card in enumerate(board_cards_str):
            board_cards[i] = rank_to_int[card[0]] + suit_to_int[card[1]]


        #Thinking round + pot + money(format this so player i comes first, then player -i) 
        stacks = state_info["Money"]
        stacks = stacks.split(' ')
        stacks = [int(x) for x in stacks]

        bet_features = [int(state_info["Round"]), int(state_info["Pot"]), stacks[current_player], stacks[1 - current_player]]

        key = np.array(hole_cards + board_cards + bet_features, dtype=np.int32)
        return key.tobytes()

    def _get_info_set(self, info_set_key, legal_actions) -> InfoSet:
        """
        Returns the information set I for the current player at a given state.
        """
        return InfoSet(info_set_key, legal_actions)

   
    def traverse(self, state: pyspiel.State, i: int, theta_1: DeepCFRModel, theta_2: DeepCFRModel,
                    M_v: MemoryReservoir, M_pi: MemoryReservoir, t: int) -> float:
        '''
        Perform a traversal of the game tree population advantage and strategy memories, recursively returning the expected payoffs of actions
        Instead of a gym environment, OpenSpiel will pass in a 'state' variable which has all the functionality of a gym environment
        '''
        # player 1 ->index 0, player 2 -> index 1
        # #print(f"Should end here {self.env.curr_round_state}")

        if state.is_terminal():
            # Terminal state get returns.
            return state.returns()[i]
        
        elif state.is_chance_node():
            # If this is a chance node, sample an action
            # Have to do this now cause we're not using the gym environment
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = np.random.choice(chance_outcome, p=chance_proba)
            return self.traverse(state.child(action), i, theta_1, theta_2, M_v, M_pi, t+1)

        current_player = state.current_player()
        #print(f"current player {current_player}, i {i}")

        # NOTE: This code will change based off of gym environment
        legal_actions = state.legal_actions()

        v_a = {} # payout for taking action
        r_Ia = {} # regret for each action

        if current_player == i:
            # Generate the info set
            info_set_key = self.get_info_set_key(state,i)

            I = self._get_info_set(
                info_set_key=info_set_key, legal_actions=legal_actions
            )

            # Compute sigma_t from the value/regret network
            if i+1 == 1:
                # Use adv_network 1
                adv_network = theta_1
            else:
                # Use adv_network 2
                adv_network = theta_2

            adv_network.eval()

            card_tensor, bet_tensor = I.convert_key_to_tensor()

            # Get predicted advantages/regrets for each action
            pred_regrets_list = adv_network(card_tensor, bet_tensor)
            pred_regrets = {}
            for idx, action in enumerate([x for x in range(NUM_ACTIONS)]):
                pred_regrets[action] = pred_regrets_list[0][idx]

            for action in I.actions():
                # NOTE: might not be =, could be +=
                I.regret[action] = pred_regrets[action]

            # Perform regret matching to compute strategy
            I.calculate_strategy()

            for action in I.actions():
            
                v_a[action] = self.traverse(state.child(action),
                    i, theta_1, theta_2, M_v, M_pi, t+1
                )

            for action in I.actions():
                expected_value = 0
                for a_prime in I.actions():
                    #print(f"sigma t {I.strategy[a_prime]}, v_a {v_a[a_prime]}")
                    expected_value += I.strategy[a_prime]*v_a[a_prime]

                # Regret
                if info_set_key not in r_Ia:
                    r_Ia[info_set_key] = {}
                r_Ia[info_set_key][action] = v_a[action] - expected_value

                # Insert the infoset and its action advantages (I t rt(I)) into the advantage memory MV
                # we need to keep track of t
                target = np.zeros(NUM_ACTIONS)

                for key, value in r_Ia[info_set_key].items():
                    target[key] = value

                card_tensor, bet_tensor = I.convert_key_to_tensor()

                # print(f"adding {torch.tensor(target)} to value memory")
                M_v.add_sample(card_tensor, bet_tensor, torch.tensor(target, dtype = torch.float32))

            #NOTE: return value not specified in paper
            return expected_value

        else:
            # Generate the info set for opposite player
            info_set_key = self.get_info_set_key(state, 1 - i)  # noam brown the goat

            I = self._get_info_set(
                info_set_key=info_set_key, legal_actions=legal_actions
            )

            if i == 1:
                # Use adv_network 1
                adv_network = theta_1
            elif i == 0:
                # Use adv_network 2
                adv_network = theta_2

            adv_network.eval()

            card_tensor, bet_tensor = I.convert_key_to_tensor()

            #print("We want this format:")
            #print(card_tensor)
            #print(bet_tensor)

            # Get predicted advantages/regrets for each action
            # NOTE: sigma_t should be list of probs
            pred_regrets_list = adv_network(card_tensor, bet_tensor)
            pred_regrets = {}
            for idx, action in enumerate([x for x in range(NUM_ACTIONS)]):
                pred_regrets[action] = pred_regrets_list[0][idx]

            for action in I.actions():
                # NOTE: might not be =, could be +=
                I.regret[action] = pred_regrets[action]

            # Perform regret matching to compute strategy
            I.calculate_strategy()

            sigma_t = [0 for i in range(NUM_ACTIONS)]

            for act in range(NUM_ACTIONS):
                sigma_t[act] = I.strategy.get(act,0)
            
            card_tensor, bet_tensor = I.convert_key_to_tensor()
            # Insert the infoset and its action probabilities (I t t(I)) into the strategy memory M_pi

            # print(f"adding { torch.tensor(sigma_t, dtype = torch.float32)} to strategy memory")
            M_pi.add_sample(card_tensor, bet_tensor, torch.tensor(sigma_t, dtype = torch.float32))

            # Sample action from strategy, check if its legal
            
            action = random.choices(list(I.strategy.keys()), weights=I.strategy.values(), k=1)[0]
            
            while action not in I.actions():
                action = random.choices(list(I.strategy.keys()), weights=I.strategy.values(), k=1)[0]


            return self.traverse(state.child(action), i, theta_1, theta_2, M_v, M_pi, t+1)

    
    def action_probabilities(self,state):
        '''
        This is a helper function that's needed for Openspiel to be capable of calculating the exploitability 
        of a given policy. If no model is passed in, function will automatically use self.policy instead
        Args:
        state: (pyspiel.State)
        Returns:
        (dict) action probabilities for a single state
        '''
        #Create the infoset object
        key = self.get_info_set_key(state, state.current_player())
        key = key = np.frombuffer(key, dtype=np.int32)
        I = self._get_info_set(key, state.legal_actions())

        strategy_network = self.policy
       
        #Use the infoset to perform inference on our strategy network, return our network's PDF over all actions
        card_tensor, bet_tensor = I.convert_key_to_tensor()
        strategy = strategy_network(card_tensor, bet_tensor)[0]      

        action_probs = {}
        
        for action in state.legal_actions():
            action_probs[action] = strategy[action].item()

        #Our neural network returns a pdf over all actions (bc NN kinda have to have a fixed output dimension)
        #So we take the probabilities of only the legal actions, and softmax it

        tensor = torch.tensor(list(action_probs.values())).to(DEVICE)
        tensor = torch.nn.functional.softmax(tensor)

        for i, action in enumerate(action_probs.keys()):
            action_probs[action] = tensor[i].item()

        return action_probs

    
    def take_action(self,state):
        '''
        Helper function I'm writing so that the agent can interface with the game_wrapper class

        Args: 
            state: (pyspiel.State)
        Returns:
            some action samples from the possible states
        '''
        pdf = self.action_probabilities(state)
        print(f"pdf {pdf}")
        values = list(pdf.keys())
        probabilities = [pdf[key] for key in values]

        # Sample from the dictionary
        action = random.choices(values, weights=probabilities, k=1) 
        return action[0]


    def compute_exploitability(self):
        return exploitability(game=self.game, policy = policy.tabular_policy_from_callable(self.game, self.action_probabilities))
    
    def _train_strategy_network(self, strategy_mem: MemoryReservoir, verbose = False) -> DeepCFRModel:
        '''
        Trains the strategy network off of the policy vectors sigma_t passed into the strategy memory during training
        and returns the newly trained neural network
        '''
         # initialize the strategy network
        strategy_network = DeepCFRModel(
            nbets=NUM_BETS, n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
        ).to(DEVICE)

        # train the strategy network to predict regrets based on infosets from the strategy memory
        dataset = strategy_mem.extract_samples()
        batch_size = BATCH_SIZE  # NOTE: In the paper, batch size of 10,000 (8192)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(strategy_network.parameters(), lr=LEARNING_RATE)

        epochs = STRATEGY_NETWORK_EPOCHS

        for epoch in range(epochs):
            strategy_network.train()  # Set model to training mode
            total_loss = 0.0
            batch = 0
            for input1, input2, output in dataloader:
                output = output.to(DEVICE)
                batch += 1

                scaler = GradScaler()

                with autocast():
                    predictions = strategy_network(input1, input2)
                    # Compute loss
                    loss = criterion(predictions, output)

                # Backward pass
                optimizer.zero_grad()  # Reset gradients

                #Using this scaler this to make training between GPUs and CPUs more efficient
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()


                # Accumulate loss
                total_loss += loss.item()

                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch:.4f}, Loss: {loss:.4f}")

        return strategy_network

    
    def train(self, iterations=1, K=10):
        '''
        This function runs the main cfr training loop, 
        iterations is the number of times a NN is trained from scratch before traversing the game tree
        K is the number of traversals per player per iteration
        '''

        log_every = iterations/log_every
        # NOTE: in the paper, they set the memory size to 40 million
        advantage_mem_1 = MemoryReservoir(max_size=MEM_SIZE)
        advantage_mem_2 = MemoryReservoir(max_size=MEM_SIZE)
        strategy_mem = MemoryReservoir(max_size=MEM_SIZE)

        # Initialize model we are training from scratch

        # Loop for `epochs` times
        for t in range(iterations):
            for i in range(self.n_players):
                # Initialize each player's value networks, and the datasets sampled from the resevior memory
                adv_network_1 = DeepCFRModel(
                    nbets=NUM_BETS, n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
                ).to(DEVICE)

                adv_network_2 = DeepCFRModel(
                    nbets=NUM_BETS, n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
                ).to(DEVICE)

                # train the advantage network to predict regrets based on infosets from the advantage memory for that player
                if i == 0:
                    dataset = advantage_mem_1.extract_samples()
                    adv_network = adv_network_1

                else:
                    dataset = advantage_mem_2.extract_samples()
                    adv_network = adv_network_2
                    
                batch_size = BATCH_SIZE

                dataloader = []
                if t > 0:
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True
                    )

                # Define a loss function with Mean Squared Error
                criterion = nn.MSELoss()

                # Define an optimizer
                optimizer = optim.Adam(adv_network.parameters(), lr=0.001)

                # Training NN from scratch
                # TODO: different epoch value ?? -> ok, so when the epoch is greater than 1, thing starts to crash xd
                epochs = 1
                for epoch in range(epochs):
                    adv_network.train()  # Set model to training mode

                    total_loss = 0.0

                    for input1, input2, output in dataloader:
                        output = output.to(DEVICE)
                        # Forward pass
                        predictions = adv_network(input1, input2)

                        # Compute loss
                        loss = criterion(predictions, output)

                        # Backward pass
                        optimizer.zero_grad()  # Reset gradients
                        loss.backward()  # Compute gradients
                        optimizer.step()  # Update model parameters

                        # Accumulate loss
                        total_loss += loss.item()

                    #print(f"Final adv mem loss: {total_loss:.4f}")

                # traverse the game tree for K iterations
                for k in range(K):
                    starting_state = self.game.new_initial_state()
                    if i == 0:
                        self.traverse(state=starting_state, i=i, 
                            theta_1=adv_network_1, theta_2=adv_network_2, M_v=advantage_mem_1, M_pi=strategy_mem, t=1
                        )
                    else:
                        self.traverse(state=starting_state, i=i, 
                            theta_1=adv_network_1, theta_2=adv_network_2, M_v=advantage_mem_2, M_pi=strategy_mem, t=1
                        )

        self.policy = self._train_strategy_network(strategy_mem=strategy_mem, verbose=True)

    def save(self, model_path='./cfr_model.pth'):
        ''' Save model
        '''
        # Get the directory from the save_path
        directory = os.path.dirname(model_path)
        
        # Check if the directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created at: {directory}")
        
        # Save the model weights
        torch.save(self.policy.state_dict(), model_path)

    #TODO: Current design of this function doesn't allow for training to continue on top of previous model weights 
    def load(self, model_path='./cfr_model.pth'):
        ''' Load model
        '''
        if not os.path.exists(model_path):
            print(f'No model found at {model_path}')
            return
        
        strategy_network = DeepCFRModel(
            nbets=NUM_BETS, n_cardstages=NUM_CARD_STAGES, n_ranks=NUM_RANKS, n_suits=NUM_SUITS, nactions=NUM_ACTIONS
        ).to(DEVICE)

        strategy_network.load_state_dict(torch.load(model_path))
        strategy_network.eval()

        self.policy = strategy_network
        
