from cfr import CFR
from deep_cfr import DeepCFR
from matplotlib import pyplot as plt
from config import *
from play_model import GameWrapper
from tqdm import tqdm
import pyspiel
from pyspiel.universal_poker import load_universal_poker_from_acpc_gamedef


def main():
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

    deep_cfr_agent = DeepCFR(game=game)
    cfr_agent = CFR(game = game)
    
    cfr_agent.load("60_it_cfr.pkl")
    # cfr_agent.train(iterations=30)
    # cfr_agent.save("30_it_cfr.pkl")

    deep_cfr_agent.load("60_it_dcfr.pth")

    poker_game = GameWrapper(agent1=deep_cfr_agent, agent2=cfr_agent, game= game, verbose=False)
    
    runs = []


    for i in range(3):

        dcfr_winnings = []
        for i in range(6):
            print("-"*10, "Iteration: ", i, "-"*10) 
            cfr_agent.train(iterations=30)
            cfr_agent.save(f"{30*(i+1)}_it_cfr.pkl")
            print("CFR agent trained")
            
            deep_cfr_agent.train(iterations=30, K = 30)
            deep_cfr_agent.save(f"{30*(i+1)}_it_dcfr.pth")
            print("Deep cfr agent trained")
            
            for i in tqdm(range(100_000)):
                poker_game.run()
            
            print(f"Deepcfr agent total winnings {poker_game.total_winnings[0]} CFR agent total winnings {poker_game.total_winnings[1]}")
            dcfr_winnings.append(poker_game.total_winnings[0])

            poker_game.total_winnings = [0,0]

        runs.append(dcfr_winnings)

    print(runs)

        

if __name__ == "__main__":
    main()