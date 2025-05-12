#!/bin/bash

# TODO: Add option to set DEBUG_ENV=true to set ros2 log-level for all nodes to DEBUG

# Functions
init_single_agent_player(){
    local player_color=$1
    local player_type=$2

    echo "Launching single-agent MATChess $2 ..."

    SINGLE_AGENT_COLOR=$player_color ros2 launch matchessbot $player_type.launch.py
}


init_multi_agent_team(){
    local player_color=$1
    local player_type="chess_piece_agent_launch_all" #$2

    echo "Launching multi-agent MATChess player $2..."

    MATCHESS_TEAM_COLOR=$player_color ros2 launch matchessbot $player_type.launch.py
}

init_matchess_manager(){

    ros2 launch matchessbot matchess_manager.launch.py
}


init_puzzle_game(){
    local puzzle_color=$1
    local solver_color=$2

    SINGLE_AGENT_COLOR=$puzzle_color ML_AGENT_COLOR=$solver_color ros2 launch matchessbot marl_vs_stockfish.launch.py
}



# init_single_agent_node $1 $2

init_puzzle_game $1 $2

