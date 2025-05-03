
import json




import math
from scipy.stats import norm
from scipy.special import erfinv

"""
References:

    https://www.chessprogramming.org/Match_Statistics
    https://3dkingdoms.com/chess/elo.htm
"""


def win_ratio(wins, losses, draws):
    """
    Calculate the win ratio of a player against an opponent.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

    Returns:

        float: The win ratio.
    """

    return (wins + draws / 2) / (wins + losses + draws)


def elo_delta_from_win_ratio(wr):
    """
    Calculate difference in Elo rating between a player and their
    opponent from the player's win ratio.

    Args:

        wr (float): The win ratio.

    Returns:

        float: The difference in Elo rating.
    """
    try:
        return 400 * math.log10(wr / (1 - wr))

    except ZeroDivisionError:
        return math.inf


def elo_delta(wins, losses, draws):
    """
    Calculate difference in Elo rating between a player and their
    opponent from the player's wins, losses, and draws.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

    Returns:

        float: The difference in Elo rating.
    """
    wr = win_ratio(wins, losses, draws)

    return elo_delta_from_win_ratio(wr)


def elo_delta_margin(wins, losses, draws, confidence=0.95):
    """
    Calculate the error margin or tolerance in Elo rating difference
    between a player and their opponent, corresponding to a given
    confidence level, from the player's wins, losses, and draws.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

        confidence (float, optional): The confidence level. Defaults to
        0.95 (95%).

    Returns:

        float: The Elo delta margin.
    """
    wr = win_ratio(wins, losses, draws)
    games = wins + losses + draws
    wins_dev = (wins / games) * math.pow(1 - wr, 2)
    draws_dev = (draws / games) * math.pow(0.5 - wr, 2)
    losses_dev = (losses / games) * math.pow(0 - wr, 2)
    std_dev = math.sqrt((wins_dev + draws_dev + losses_dev) / games)

    min_confidence = (1 - confidence) / 2
    max_confidence = 1 - min_confidence
    min_dev = wr + std_dev * norm.ppf(min_confidence)
    max_dev = min(wr + std_dev * norm.ppf(max_confidence), 0.999)

    margin = (elo_delta_from_win_ratio(max_dev) - elo_delta_from_win_ratio(min_dev)) / 2

    return margin


def likelihood_of_superiority(wins, losses):
    """
    Calculate the likelihood of superiority (LOS) of a player over their
    opponent, from their wins and losses.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

    Returns:

        float: The likelihood of superiority.
    """

    return 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * (wins + losses))))




if __name__ == '__main__':

    # Test data files:
    test_result_dir = "./test_results/compare_methods_to_fairystockfish/"
    astar_vs_random = "performance_eval_astar_vs_random_1741022631544790343.json"
    astar_vs_fairy = "performance_eval_astar_vs_fairystockfish1741014157775394594.json"

    decformer_vs_random = "performance_eval_decformer_vs_random_1741021490536380740.json"
    decformer_vs_fairy = "performance_eval_decformer_vs_fairystockfish1741009200792001183.json"

    encformer_vs_random = "performance_eval_encformer_vs_random_1741022476745852505.json"
    encformer_vs_fairy = "performance_eval_encformer_vs_fairystockfish1741012316957436508.json"

    files_dict = {
        'astar_vs_random': astar_vs_random,
        'astar_vs_fairy': astar_vs_fairy,
        'decformer_vs_random': decformer_vs_random,
        'decformer_vs_fairy': decformer_vs_fairy,
        'encformer_vs_random': encformer_vs_random,
        'encformer_vs_fairy': encformer_vs_fairy
    }

    # win_stats_astar_vs_random = {'win': 0, 'loss': 0, 'draw': 0}
    # win_stats_astar_vs_fairy = {'win': 0, 'loss': 0, 'draw': 0}

    # win_stats_decformer_vs_random = {'win': 0, 'loss': 0, 'draw': 0}
    # win_stats_decformer_vs_fairy = {'win': 0, 'loss': 0, 'draw': 0}

    # win_stats_encformer_vs_random = {'win': 0, 'loss': 0, 'draw': 0}
    # win_stats_encformer_vs_fairy = {'win': 0, 'loss': 0, 'draw': 0}
    
    # win_stats_dict = {
    #     'astar_vs_random': {'win': 0, 'loss': 0, 'draw': 0},
    #     'astar_vs_fairy': {'win': 0, 'loss': 0, 'draw': 0},
    #     'decformer_vs_random': {'win': 0, 'loss': 0, 'draw': 0},
    #     'decformer_vs_fairy': {'win': 0, 'loss': 0, 'draw': 0},
    #     'encformer_vs_random': {'win': 0, 'loss': 0, 'draw': 0},
    #     'encformer_vs_fairy': {'win': 0, 'loss': 0, 'draw': 0}
    # }
    win_stats_dict = {
        'astar-2': {'random': {'win': 0, 'loss': 0, 'draw': 0}, 'fairystockfish-elo500': {'win': 0, 'loss': 0, 'draw': 0}},
        'decformer': {'random': {'win': 0, 'loss': 0, 'draw': 0}, 'fairystockfish-elo500': {'win': 0, 'loss': 0, 'draw': 0}},
        'encformer': {'random': {'win': 0, 'loss': 0, 'draw': 0}, 'fairystockfish-elo500': {'win': 0, 'loss': 0, 'draw': 0}},
    }
    player_names = ["astar-2", "decformer", "encformer"]
    opponent_names = ["random", "fairystockfish-elo500"]


    for key, filename in files_dict.items():
        with open(test_result_dir + filename, 'r') as infile:
            for line in infile.readlines():
                eval_game_dict = json.loads(line)

                method_is_white = None
                method_name = None
                opponent_name = None
                if eval_game_dict['player_white'] in player_names:
                    method_is_white = True
                    method_name = eval_game_dict['player_white']
                    opponent_name = eval_game_dict['player_black']
                else:
                    method_is_white = False
                    method_name = eval_game_dict['player_black']
                    opponent_name = eval_game_dict['player_white']
                
                if eval_game_dict['game_result'] == "1-0":
                    if method_is_white:
                        win_stats_dict[method_name][opponent_name]['win'] += 1
                    else:
                        win_stats_dict[method_name][opponent_name]['loss'] += 1
                elif eval_game_dict['game_result'] == "0-1":
                    if method_is_white:
                        win_stats_dict[method_name][opponent_name]['loss'] += 1
                    else:
                        win_stats_dict[method_name][opponent_name]['win'] += 1
                else:
                    win_stats_dict[method_name][opponent_name]['draw'] += 1
            infile.close()
    
    
    
    # print(win_stats_dict)
    for key, item in win_stats_dict.items():
        print(f"\n\nPlayer:\t{key}")
        print(f"Stats:\n\t{item}")
        for opponent_name, game_result_stats in item.items():
            method_win_ratio = win_ratio(wins=float(game_result_stats['win']), losses=float(game_result_stats['loss']), draws=float(game_result_stats['draw']))
            print(f"{key} win ratio against {opponent_name}:\t{method_win_ratio}")

    





    
    




