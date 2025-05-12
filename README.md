# README

## Setup ROS2 humble workspace
1. Create a ROS2 workspace and clone this repo into the `src/` dir:
```
cd ros2ws/src/
git clone git@github.com:SkovIda/multiagent_project.git
```
2. install dependencies (cyclonedds is not in rosdep, therefore '-r')
```
rosdep install --from-paths src/matchessbot/ --ignore-src -y --rosdistro=humble -r
```
3. Install requirements from workspace root dir: `ros2ws/`:
    - 
    ```
    pip3 install -r src/matchessbot/requirements.txt
    ```
    - NOTE: Don't use virtualenv. ros2 humble does not mix well with virtual envs. See cautionary note about virtual envs ROS2 in docs: https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html#using-python-packages-with-ros-2
    - NOTE: need to install pytorch with pip because `python3-torch 1.8.1-4 (amd64 binary) in ubuntu jammy` (ubuntu 22.04) is a CPU-only version of PyTorch: https://launchpad.net/ubuntu/jammy/amd64/python3-torch/1.8.1-4

## Build the packages
1. Use `colcon` to build the packages from workspace root dir `ros2ws/`: `colcon build --packages-select matchessbot matchess_interfaces`

## Run MATChess game with 16 chess piece agents vs. various opponents:
<!-- 1. Build the matchess interfaces package from workspace root dir `ros2ws/`: `colcon build --packages-select matchess_interfaces`
1. Build matchessbot package from workspace root dir `ros2ws/`: `colcon build --packages-select matchessbot` -->
1. Source the workspace (from `ros2ws/` in terminal): `source install/setup.bash`
1. set ros middleware layer to use cyclonedds: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (fastdds loses messages with reliable QoS profile)
1. Launch the MATChess Game Manager and a team of 16 chess piece agents:
    - Launch the MATChess Game Manager and all chess piece agents on the:
        - white team: `ros2 launch matchessbot matchess.launch.py`
        - black team: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot matchess.launch.py`
    - Or launch the Matchess Game Manager and a team of chess piece agents seperately:
        1. Run the MATChess Game Manager: `ros2 launch matchessbot matchess_manager.launch.py`
        1. Launch all 16 pieces on a team:
            - All white pieces: `MATCHESS_TEAM_COLOR=white ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
            - All black pieces: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
1. Choose one of the following options for running the node/nodes that acts as the opposing player:
    1. Launch one of the following single-agent opponents (it uses the Stockfish engine to select its moves):
        - Stockfish engine playing as white: `SINGLE_AGENT_COLOR=white ros2 launch matchessbot player_stockfish.launch.py`
        - Stockfish engine playing as black: `SINGLE_AGENT_COLOR=black ros2 launch matchessbot player_stockfish.launch.py`
    1. Launch another team of 16 chess piece agents:
        - All white pieces: `MATCHESS_TEAM_COLOR=white ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
        - All black pieces: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
    1. Launch the multi-agent machine learning model from a "single-agent" chess player, where the model is used to obtain the MoveVotes from each of the 16 individual chess piece agents on the team, collect and count those votes, before sending a single MoveVote to the manager with the result:
        - The team of white pieces:`ML_AGENT_COLOR=white ros2 launch matchessbot player_matchess.launch.py`
        - The team of black pieces: `ML_AGENT_COLOR=black ros2 launch matchessbot player_matchess.launch.py`
1. Init new game:
    <!-- `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/out matchess_interfaces/msg/ChessMove "uci: ''" --once --qos-reliability reliable` -->
    1. source the workspace: `source ros2ws/install/setup.bash`
    1. Reset the state of the game to the standard chess starting position for all agents in the game:
        ```
        RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'RESET_GAME_STATE',status_int: 1}" --once --qos-reliability reliable --qos-durability transient_local
        ```
        <!-- ```
        RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/game_status_cmd matchess_interfaces/msg/GameStatus "{status_str: 'RESET_GAME_STATE',status_int: 1}" --once --qos-reliability reliable --qos-durability transient_local
        ``` -->
    1. Optional: Setup the state of the game from a MoveHist before starting the game:
        1. Pub message to put all nodes in "setup mode":
            ```
            RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'SET_GAME_STATE_FROM_HIST',status_int: 8}" --once --qos-reliability reliable --qos-durability transient_local
            ```
        1. Set the game state of all the agents in the game from a list of UCI moves. As an example, The following command will set the state of all the agents in the game to the state AFTER white has made their first move: "e2e4":
            ```
            RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameHist "{move_hist_uci: ["e2e4"]}" --once --qos-reliability reliable --qos-durability transient_local
            ```

    1. Tell the agents to start the game from the current game state of the agents
        <!-- ```
        RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/game_status_cmd matchess_interfaces/msg/GameStatus "{status_str: 'START_GAME',status_int: 2}" --once --qos-reliability reliable --qos-durability transient_local
        ``` -->
        ```
        RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/ui_cli matchess_interfaces/msg/GameStatus "{status_str: 'START_GAME',status_int: 2}" --once --qos-reliability reliable --qos-durability transient_local
        ```
    

## Run MATChess in puzzle mode:
1. Setup
    ```
    cd ros2ws/
    source install/setup.bash
    ```
1. run matchess in puzzle mode:
    ```
    ./src/matchessbot/scripts/run_matchess_puzzles.sh "<puzzle_idx>"
    ```
    - NOTE: Command above will run the manager node, the ML agent node (player using the Machine Learning model to choose its next move), and the puzzle node (player using the stockfish engine to choose its next move)

    <!-- ```
        cd src/matchessbot/scripts/
        ./run_matchess.sh white black
        ```-->
    <!-- - The command above is equivalent to:
        ```
        SINGLE_AGENT_COLOR=white ML_AGENT_COLOR=black ros2 launch matchessbot marl_vs_stockfish.launch.py
        ```  -->



## Implemented Program Currently has the Following Features:
- Each agent (all the chess_piece_agents in the MAS and the *single-agent opponent player*) knows the complete state of the game at start up.
    - All 16 pieces on a team can be launced with a single command.
    - All pieces have a unique ID: The name of the type of chess piece and the index of the file corresponding to the square that the chess_piece_agent is placed on at the start of a new game with the standard setup for chess.
    - On their turn, the agents publish a "move vote" (for the current implementation, the chess_piece_agents just randomly selects a legal move for the current state of the game).
- MatchessManager:
    - Handles collecting and counting the votes that all the chess piece agents publish. Once all agents on a team has broadcasted their prefered next move, the MatchessManager publishes the move with the most votes to all agents in the game.
    - Logs the half-moves that each player performed on their turn, which move each agent voted for on each of their turns, as well as images of the current state of the game (`.png` files with 2D renderings of a chess board).
- PlayerStockfish:
    - The player_stockfish agent votes for moves just like the team of chess_piece_agents do. Since player_stockfish agent is a single agent, who controls all the pieces of one color, it only casts one vote per turn, which automatically wins the vote. The Matchess_manager handles this by considering the player_stockfish agent as a "team consisting of a single agent" and collects the votes from "all the agents on the team" (which is one for this player) before counting them the same way it does for a team consisting of multiple agents.


## TODO
- [X] Load chess engine in player_stockfish from file path in config:
    1. [X] Add engine path to ros-params in config files: `config/white_single_agent_params.yaml` and `config/white_single_agent_params.yaml`
    1. [X] Load engine from that path in player_stockfish instead of hardcoded path
- [X] Make the manager control the overall game status of the nodes, such that game state of the nodes are controlled from the matchess_manager:
    - The *matchess_manager* should be able to send commands to the *chess_piece_agent* to change its state:
        - GAME_STATUS.SET_GAME_STATE_FROM_HIST, GAME_STATUS.START_GAME, GAME_STATUS.KILL_NODE, GAME_STATUS.RESET_GAME_STATE
    - The *chess_piece_agent* nodes should be able to change its own game_status to one of the following:
        - GAME_STATUS.NONE, GAME_STATUS.IDLE, GAME_STATUS.READY_TO_PLAY, GAME_STATUS.GAME_IN_PROGRESS, GAME_STATUS.GAME_OVER
- [X] Debug code that setup new game from a later stage of a game, which is set up from a list of UCI moves:
    - [X] Uses a GAME_STATUS command used to set the agent's game states from a list of UCI moves that was played to reach it: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/game_status_cmd matchess_interfaces/msg/GameStatus "{status_str: 'SET_GAME_STATE_FROM_HIST',status_int: 8}" --once --qos-reliability reliable --qos-durability transient_local`
    - [X] Use CLI to input a GameHist to matchess_manager that will be used to set up the game state instead of the temporary hardcoded GAME_HIST in the ui_cli_callback() in the matchess_manager node!
- [ ] Make a general single-agent player for running different chess engines with the same single-agent player node:
    1. [ ] Create a new chess player node class: chess_player_agent.py (same functionality as the player_stockfish node except for the following changes)
    1. [ ] Make a new launch file: `single_agent_player.launch.py`
        - [ ] Same as `player_stockfish.launch.py` but has an additional environment variable: `SINGLE_AGENT_PLAYER_TYPE`
        - [ ] Use this environment variable to load differnt chess engines, which will be passed as a parameter to the `chess_player_agent.py` to the single-agent player
    1. [ ] Add `engine_path` parameter to node and config files: `config/white_single_agent_params.yaml` and `config/white_single_agent_params.yaml`
    1. [ ] use value of `engine_path` param to load chess engine in node
- [ ] Add feature for evaluating performance of the MAS with various chess puzzles
- [ ] Add visualization to the framework:
    1. pub game state and voting round info from the matchess_manager node to `matchess/visualization`
    1. pub game state and "decision-making info" (i.e. votes and network) of the chess piece agents
