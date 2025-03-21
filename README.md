# README

## Setup ROS2 humble workspace
1. Create a ROS2 workspace and clone this repo into the `src/` dir:
```
cd ros2ws/src/
git clone git@github.com:SkovIda/multiagent_project.git
```
<!-- 1. Set environment variable: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` -->

## Run chess game with MATChess bot (white) vs. Stockfish engine (black):
1. Build the matchess interfaces package from workspace root dir `ros2ws/`: `colcon build --packages-select matchess_interfaces`
1. Build matchessbot package from workspace root dir `ros2ws/`: `colcon build --packages-select matchessbot`
1. Source the workspace (from `ros2ws/` in terminal): `source install/setup.bash`
1. set ros middleware layer to use cyclonedds: `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (fastdds loses messages with reliable QoS profile)
1. Run the MATChess Game Manager: `ros2 launch matchessbot matchess_manager.launch.py`
1. Launch all 16 pieces on a team:
    - All white pieces: `MATCHESS_TEAM_COLOR=white ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
    - All black pieces: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
1. Run the node that acts as the opposing player (it uses the Stockfish engine to select its moves and is currently hardcoded to always play as black):
    1. Stockfish engine player: `ros2 run matchessbot player_stockfish`
1. Init new game: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ros2 topic pub /matchess/out matchess_interfaces/msg/ChessMove "uci: ''" --once --qos-reliability reliable`


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
- [X] Shutdown the chess piece agent (which is a single node) when it knows that the game is over:
    1. [X] Add publisher to chess_piece_agent node that publishes to a "general system status" topic. 
        - [X] Create a *GameStatus.msg* in the "matchess_interfaces" package. Should contain an int and a string representing the *game status code*
    1. [X] Add a *main logic* callback to the chess_piece_agent node (which is just a timer_callback function). This function is the main loop for the actual robot. It should contain the following:
        1. [X] Comment that says: "main robot logic goes here". (NOTE: the main robot logic is the control loop (motor control), make robot move to "home position" on start up, what the robot should do before shutdown, etc.)
        1. [X] Publish a message to the *"general system status" topic:* 'matchess/game_status' with the overall status of the game (from the agent's perspective): NONE, OK, STOP
        1. [X] Raise a *shutdown node error* in *main logic* callback when the *game-over flag* is set and catch this error in the main loop with a "try catch" around the `rclpy.spin(node_name)` in the main function
        1. [ ] *main logic* function should also handle logging/publishing some info about the current game / state of the agent/robot at each step
- [X] Shutdown the matchess_manager node by listening to the *'matchess/game_status' topic* and catch the *shutdown node error* in the main loop with a "try except" around the `rclpy.spin(node_name)`
- [X] Shut down the player_stockfish node just like the chess piece agents are shut down: pub msg to the *'matchess/game_status' topic* and catch the *shutdown node error* in the main loop with a "try except" around the `rclpy.spin(node_name)`