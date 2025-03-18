# README

<!-- ## Setup ROS2 humble workspace
1. clone content of `src/` dir into the workspace: `ros2ws/src` dir: <mark>INSERT COMMAND TO CLONE GIT REPO!!!</mark>

## Run MATChess bot in terminal
1. Build package from workspace root dir `ros2ws/`: `colcon build --packages-select matchessbot`
1. source the workspace (from `ros2ws/` in terminal): `source install/setup.bash`
1. Launch all 16 pieces on a team:
    - All white pieces: `MATCHESS_TEAM_COLOR=white ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
    - All black pieces: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
1. Example of how to publish a chess move in uci notation to the input topic that matchessbot subscribes to: 
    1. Open a new terminal and source in the workspace: `source ros2ws/install/setup.bash`
    1. Publish uci move to `/matchess/out` topic: `ros2 topic pub -1 /matchess/out std_msgs/msg/String "data: e2e4"`
    - **NOTE:** Examine how to ensure that the message is published such that all agents recieve it?
1. Example of using the MATChess Game Manager (work in progress):
    1. Build the MatchessManager package from workspace root dir `ros2ws`: `colcon build --packages-select matchess_interfaces`
    1. Open a new terminal and source in the workspace: `source install/setup.bash`
    1. run following command in terminal: `ros2 run matchessbot matchess_manager`
    1. Currently only publishes a uci chess move to the `matchessbot/out` topic once (NOTE: this move will NOT be recieved by all 16 agents in the MAS!!!)


<!-- ## Init a new game with a team of white pieces against a random opponent (work in progress)
1. source the workspace in 3 different terminals by running the following command from workspace root dir `ros2ws/`: `source install/setup.bash`
1. Each of the following commands can then be run in either of those 3 terminals:
    1. launch <mark>white king and queen chess_piece_agents</mark>: `ros2 launch chess_piece chess_piece_agents_launch.py`
    1. Start the Matchess environment node: `ros2 run matchess talker`
    1. Publish a message that requests matchess_env to publish the current state of the game: `ros2 topic pub /matchess/input std_msgs/msg/String "data: ReqObs"` 

## TODO
- [x] create all 16 chess piece agents (how to do this properly?)
- [x] Fix launch config files for black and white pieces to not have prefix: "black_" and "white_"
- [x] Update readme with *how to launch a team of either black or white pieces*
- [x] implement custom interfaces for messages with moves?
    - **NOTE:** Just made a custom interface for transmitting a uci string
- [ ] implement most simple version of a voting system for all agents on a team
    - [ ] make agents coose random moves to vote for
    - [ ] make agent that needs to move, publish the chosen move to the "matchess/in" topic
    - [ ] play game against "Random MATChessBot" via pub to *matchess topics* from command line
- [ ] make a script for playing a game of chess between the team and a random opponent?
    - **NOTE:** in progress of implementing this functionality in matchess_manager.py
- [ ] setup custom reward functions in matchess

# README (v2) -->

## Setup ROS2 humble workspace
1. Create a ROS2 workspace and clone this repo into the `src/` dir: <mark>INSERT COMMAND TO CLONE GIT REPO!!!</mark>


## Run chess game with MATChess bot (white) vs. Stockfish engine (black):
1. Build the matchess interfaces package from workspace root dir `ros2ws`: `colcon build --packages-select matchess_interfaces`
1. Build matchessbot package from workspace root dir `ros2ws/`: `colcon build --packages-select matchessbot`
1. Source the workspace (from `ros2ws/` in terminal): `source install/setup.bash`
1. Run the MATChess Game Manager:
    1. Open a new terminal and source the workspace (from workspace root dir `ros2ws/`): `source install/setup.bash`
    1. run the following command in a terminal to start the MatchessManager node: `ros2 run matchessbot matchess_manager`
1. Launch all 16 pieces on a team:
    - All white pieces: `MATCHESS_TEAM_COLOR=white ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
    - All black pieces: `MATCHESS_TEAM_COLOR=black ros2 launch matchessbot chess_piece_agent_launch_all.launch.py`
1. Run the node that acts as the opposing player in a new terminal (it uses the Stockfish engine to select its moves and is currently hardcoded to always play as black???):
    1. Open a new terminal and source in the workspace: `source ros2ws/install/setup.bash`
    1. Run the stockfish engine player: `ros2 run matchessbot player_stockfish`


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
- [ ] Shutdown the chess piece agent (which is a single node) when it knows that the game is over:
    1. [ ] Add publisher to chess_piece_agent node that publishes to a "general system status" topic. <mark>NOTE: This topic must only be used for big system changes / setup messages (like node shutdown due to game over and more???)</mark>
        - [ ] Create a *RobotSimStatus.msg* to the "matchess_interfaces" package. Should contain an int representing a *robot status code*
    1. [ ] Set a *game-over flag* in the listener callback function. This flag will be used to shut down the node in the *main robot logic callback* (i.e. timer_callback function)
    1. [ ] Add a *main robot logic callback* (which is just a timer_callback function) to the chess_piece_agent node. This function is the main loop for the actual robot. It should contain the following:
        1. [ ] Comment that says: "main robot logic goes here". (NOTE: the main robot logic is the control loop (motor control), make robot move to "home position" on start up, what the robot should do before shutdown, etc.)
        1. [ ] Publish a message to the *"general system status" topic* with the overall status of the robot
            - What status can the chess_piece_agent have?
            - <mark>TODO: Ask Stefan how this message should be used by the matchess manager to handle shutting that node down as well</mark>
        1. [ ] Raise a *shutdown node error* in *main robot logic callback* when the *game-over flag* is set and catch this error in the main loop with a "try catch" around the `rclpy.spin(node_name)`
        1. <mark>*main robot logic callback* function should also handle logging/publishing some info about the current game / state of the agent/robot at each step?</mark>
    1. [ ] Catch the *shutdown node error* in the main loop with a "try except" around the `rclpy.spin(node_name)`
- [ ] Shutdown the matchess_manager and the player_stockfish nodes by listening to the *"general system status" topic* or ??? <mark>TODO: Ask Stefan</mark>
<!-- 1. [ ] Shutdown node by raising an error when the *game-over flag* is set? <mark>Should do some clean up before shutdown???</mark> -->

**NOTE:** Shutting down all the nodes properly after the game is over should fix the following error (which is currently present in the program): The chess_piece_agents continue to publish a move (which is just an empty string) after they lost the game. The matchess_manager incorrectly collects these "empty votes" as movevotes and attempts to apply it move with the most votes to the board, which is an illegal move and the node crashes.
