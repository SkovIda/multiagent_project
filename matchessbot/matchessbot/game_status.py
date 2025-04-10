from enum import Enum

class GAME_STATUS(Enum):
    NONE = -1
    # OK = 0
    STOP = 0
    
    # Game status controlled/set by the MatchessManager:
    RESET_GAME_STATE = 1
    START_GAME = 2
    KILL_NODE = 3
    SET_GAME_STATE_FROM_HIST = 8

    # Game status controlled/set by the agent:
    IDLE = 4
    READY_TO_PLAY = 5
    GAME_IN_PROGRESS = 6
    GAME_OVER = 7
    
