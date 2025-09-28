import pyspiel
from pyspiel.universal_poker import load_universal_poker_from_acpc_gamedef
from config import *

import argparse
from cfr import CFR
from human_player import Human_Player

import numpy as np

class GameWrapper:
    def __init__(self, game, agent1, agent2, verbose: bool = True):
        self.game = game
        self.agent1 = agent1
        self.agent2 = agent2
        self.total_winnings = [0,0]
        self.verbose = verbose

        self.state = game.new_initial_state()

    def run(self):
        while(not self.state.is_terminal()):

            if(self.state.is_chance_node()):
                chance_outcome, chance_proba = zip(*self.state.chance_outcomes())
                action = np.random.choice(chance_outcome, p=chance_proba)
                self.state = self.state.child(action)

            elif(self.state.current_player() == 0):
                action = self.agent1.take_action(self.state)
                if self.verbose: print(action)
                self.state = self.state.child(action)
            else:
                self.state = self.state.child(self.agent2.take_action(self.state))

        if self.verbose: print(f"Hand ended at state: \n {self.state}")

        self.total_winnings = [self.total_winnings[i] + self.state.returns()[i] for i in range(2)]

        if self.verbose: print(f"Winnings for that hand are {self.state.returns()}")
        if self.verbose: print(f"Total winnings are {self.total_winnings}")
        if self.verbose: print("Press any key to continue")

        self.state = self.game.new_initial_state()

        if self.verbose: input()


def main(args):
     # Load the game environment 

    poker_variant =f"""\
GAMEDEF
nolimit
numPlayers = 2
numRounds = {NUM_CARD_STAGES}
blind = {SMALL_BLIND} {BIG_BLIND}
maxRaises = {MAX_RAISES} {MAX_RAISES}
numSuits = {NUM_SUITS}
numRanks = {NUM_RANKS}
numHoleCards = {NUM_HOLE_CARDS}
numBoardCards = {" ".join(map(str, BOARD_CARDS))}
bettingAbstraction = {BETTING_ABSTRACTION}
END GAMEDEF
"""
    game = load_universal_poker_from_acpc_gamedef(poker_variant)

    human= Human_Player()
    pokerbot= CFR(game=game)
    pokerbot.load(args.opponent_model)

    poker = GameWrapper(game=game, agent1=human, agent2=pokerbot)

    for i in range(args.num_hands):
        poker.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--opponent_model',
        type=str,
        default='./cfr_model(300).pth',
    )

    parser.add_argument(
        "--num_hands",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    main(args)