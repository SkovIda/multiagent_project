#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# PUZZLE_DATASET_NAME="puzzles.csv"
PUZZLE_DATASET_NAME=$2 #"puzzles_checkmate_in_2_max_elo_400.csv"

TEST_TIMESTAMP=$3 #"$EPOCHSECONDS"

init_puzzle_nodes(){
    local puzzles_csv="$SCRIPT_DIR/../tools/$PUZZLE_DATASET_NAME"

    local puzzle_id=$(python3 $SCRIPT_DIR/../tools/read_csv_row.py --csv_file $puzzles_csv --row_number $1 --result 'puzzle_id')
    echo "puzzle_id: $puzzle_id"

    local puzzle_color=$(python3 $SCRIPT_DIR/../tools/read_csv_row.py --csv_file $puzzles_csv --row_number $1 --result 'puzzle_color')
    echo "puzzle_color: $puzzle_color"

    local solver_color=$(python3 $SCRIPT_DIR/../tools/read_csv_row.py --csv_file $puzzles_csv --row_number $1 --result 'player_color')
    echo "solver_color: $solver_color"

    MATCHESS_DEBUG=false TEST_ID=$TEST_TIMESTAMP PUZZLE_ID=$puzzle_id SINGLE_AGENT_COLOR=$puzzle_color ML_AGENT_COLOR=$solver_color ros2 launch matchessbot marl_vs_stockfish.launch.py 2>&1 &
}


reset_game_state(){
    echo "Setting up the game..."
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'RESET_GAME_STATE',status_int: 1}" --once --qos-reliability reliable --qos-durability transient_local
}


# setup_puzzle_from_hist(){
#     local uci_move_hist=["e2e4"]

#     echo "Setting game state from hist $uci_move_hist"

#     sleep 0.1

#     RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'SET_GAME_STATE_FROM_HIST',status_int: 8}" --once --qos-reliability reliable --qos-durability transient_local

#     sleep 2

#     RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameHist "{move_hist_uci: $uci_move_hist}" --once --qos-reliability reliable --qos-durability transient_local

#     # echo "Done setting game state from hist."
# }

setup_puzzle_from_line_idx(){
    echo "SCRIPT_DIR: $SCRIPT_DIR" #$(sed "50000000q;d" myfile.ascii) 2>&1

    # local puzzles_csv="$SCRIPT_DIR/../tools/puzzles.csv"
    local puzzles_csv="$SCRIPT_DIR/../tools/$PUZZLE_DATASET_NAME"
    
    local uci_move_hist=$(python3 $SCRIPT_DIR/../tools/read_csv_row.py --csv_file $puzzles_csv --row_number $1 --result 'move_hist')

    echo "Setting game state from hist $uci_move_hist"
    sleep 0.1
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'SET_GAME_STATE_FROM_HIST',status_int: 8}" --once --qos-reliability reliable --qos-durability transient_local
    sleep 2
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameHist "{move_hist_uci: $uci_move_hist}" --once --qos-reliability reliable --qos-durability transient_local

    echo "Done setting game state from hist."
}


start_game(){
    echo "Start Game"
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'START_GAME',status_int: 2}" --once --qos-reliability reliable --qos-durability transient_local
}


# for i in {1..10};
# do
#     # echo "Iteration $i"
#     # Start the nodes:
#     echo "====================\nMATChess: Puzzle Mode\n===================="
#     echo "Waiting for all nodes to init..."

#     init_puzzle_nodes $i
#     sleep 1

#     # Init new MATCHess game:
#     reset_game_state
#     sleep 1

#     setup_puzzle_from_line_idx $i

#     sleep 2

#     start_game

#     sleep 0.1
#     echo "Waiting for game to finish..."
#     # Wait for all jobs to finish:
#     wait $(jobs -p)
#     echo "All processes finished"

# done


##### Load puzzle from its index in puzzles.csv file:
# source install/setup.bash

# Start the nodes:
echo "==================== MATChess: Puzzle Mode ===================="
echo "Waiting for all nodes to init..."

init_puzzle_nodes $1
sleep 1

# Init new MATCHess game:
reset_game_state
sleep 1

setup_puzzle_from_line_idx $1

sleep 4

start_game

sleep 1
echo "Waiting for game to finish..."
# Wait for all jobs to finish:
wait $(jobs -p)
echo "All processes finished"

