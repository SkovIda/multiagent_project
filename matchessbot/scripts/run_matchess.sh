#!/bin/bash

# TODO: Add option to set DEBUG_ENV=true to set ros2 log-level for all nodes to DEBUG

# # Functions
# init_single_agent_player(){
#     local player_color=$1
#     local player_type=$2

#     echo "Launching single-agent MATChess $2 ..."

#     SINGLE_AGENT_COLOR=$player_color ros2 launch matchessbot $player_type.launch.py
# }


# init_multi_agent_team(){
#     local player_color=$1
#     local player_type="chess_piece_agent_launch_all" #$2

#     echo "Launching multi-agent MATChess player $2..."

#     MATCHESS_TEAM_COLOR=$player_color ros2 launch matchessbot $player_type.launch.py
# }

# init_matchess_manager(){

#     ros2 launch matchessbot matchess_manager.launch.py
# }


init_puzzle_nodes(){
    local puzzle_color=$1
    local solver_color=$2

    SINGLE_AGENT_COLOR=$puzzle_color ML_AGENT_COLOR=$solver_color ros2 launch matchessbot marl_vs_stockfish.launch.py 2>&1 &
}


reset_game_state(){
    echo "Setting up the game..."
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'RESET_GAME_STATE',status_int: 1}" --once --qos-reliability reliable --qos-durability transient_local
}


setup_puzzle_from_hist(){
    # local uci_move_hist="['e2e4']" #$(python chess_notation_converter.py --san_str "$1")
    local uci_move_hist=["e2e4"]
    echo "Setting game state from hist $uci_move_hist"

    sleep 0.1

    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'SET_GAME_STATE_FROM_HIST',status_int: 8}" --once --qos-reliability reliable --qos-durability transient_local

    sleep 2

    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameHist "{move_hist_uci: $uci_move_hist}" --once --qos-reliability reliable --qos-durability transient_local

    # echo "Done setting game state from hist."
}


start_game(){
    echo "Start Game"
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'START_GAME',status_int: 2}" --once --qos-reliability reliable --qos-durability transient_local
}


# init_single_agent_node $1 $2

# Start the nodes:
echo "====================\nMATChess: Puzzle Mode\n===================="
echo "Waiting for all nodes to init..."

init_puzzle_nodes $1 $2
sleep 1

# Init new MATCHess game:
reset_game_state
sleep 1

setup_puzzle_from_hist $3

sleep 2

start_game

sleep 0.1
echo "Waiting for game to finish..."
# Wait for all jobs to finish:
wait $(jobs -p)
echo "All processes finished"
