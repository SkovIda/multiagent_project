# from .pettingzoo_chess import env, raw_env
# import chess_utils
# # from pettingzoo.classic import chess_v6


# # class MATChess:
# #     def __init__(self):
# #         self.chess_env = env(render_mode="ansi")
# #         self.chess_env.reset(seed=42)
    
# #     def init_game(self):
# #         self.chess_env.reset(seed=42)   # TODO: add load game params from config here and init the game (i.e., pub game state and wait for msg about the chosen move?)

# #         # Init variable storing itr of the active player:
# #         self.active_player = self.chess_env.agent_iter()    # TODO: check if this is how it should be done!

# #         # for agent in self.chess_env.agent_iter():
# #         #     observation, reward, termination, truncation, info = self.chess_env.last()

# #         #     if termination or truncation:
# #         #         action = None
# #         #     else:
# #         #         mask = observation["action_mask"]
# #         #         # this is where you would insert your policy
# #         #         action = self.chess_env.action_space(agent).sample(mask)

# #         #     self.chess_env.step(action)
# #         # self.chess_env.close()
    
# #     def environment_step(self):
# #         # TODO: apply MAS's action and step env in this func for both the "MAS player"  and the opponent player. Will be called by the MATChess_action_sub node when MAS has send msg with their next action
# #         pass

    
# #     def end_game(self):
# #         self.chess_env.close()

# def run_game(env):
#     # from pettingzoo.classic import chess_v6

#     # env = chess_v6.env(render_mode="human")
#     env.reset(seed=42)

#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()

#         if termination or truncation:
#             action = None
#         else:
#             mask = observation["action_mask"]
#             # this is where you would insert your policy
#             action = env.action_space(agent).sample(mask)

#         env.step(action)
#     env.close()



# if __name__ == '__main__':
#     chess_env = env(render_mode="human")
#     chess_env.reset(seed=42)
#     # matches_game = MATChess()



#     # for agent in chess_env.agent_iter():
#     #     observation, reward, termination, truncation, info = chess_env.last()

#     #     if termination or truncation:
#     #         action = None
#     #     else:
#     #         mask = observation["action_mask"]
#     #         # this is where you would insert your policy
#     #         action = chess_env.action_space(agent).sample(mask)

#     #     chess_env.step(action)
#     # chess_env.close()
