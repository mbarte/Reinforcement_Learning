# Reinforcement_Learning
A reinforcement learning problem involving chess.

By me ([@mbarte](https://github.com/mbarte)) and [@GuidoGiacomoMussini](https://github.com/GuidoGiacomoMussini).
_______________________________________
In this project we implemented from scratch a small chess engine (white agent), which is able to play and win the final in the picture below against Stockfish 15.1.

<p align="center">
<img src="https://user-images.githubusercontent.com/99347573/233844371-3f940bc0-4f9b-43b6-9ec8-156dc5d5431e.png" width="300" height="300">
<p align="center">
Starting position.
</p>

In order for a user to run the code, he should download and install Stockfish.

<p align="center">
<img src="https://user-images.githubusercontent.com/99347573/233845204-c8625152-71ba-4dcc-809d-fc44abb55d39.png" width="325" height="300">
<p align="center">
Game results trend.
</p>

_______________________________________

- **dynamics_br**: implementation of the "backbone" of the game from scratch (dictionaries to define pieces and their moves, functions to retrieve possible moves and allowed future states, move pieces, etc.).

- **algorithm_utils_br**: functions used by the learning algorithm scarsa_lambda_br.

- **scarsa_lambda_br**: learning algorithm (sarsa-lambda, with epsilon-greedy approach).

- **executable_br**: Jupyter Notebook to train the agent.

- **user_vs_scarsa_lambda_br**: adaptation of scarsa_lambda_br replacing Stockfish with input from keyboard to allow users to play against our trained agent (one should download the Q_s_20t_1000g.pickle file, containing the "parameters" of the trained agent).

- **game_br**:  Jupyter Notebook to play against our trained agent.

________________________________________

- **Q_s_20t_1000g**: Dictionary with the trained agent's parameters.

- **stockfish_15.1_win_x64_popcnt**: folder containing Stockfish executable.
_______________________________________
Future work: longer training against Stockfish with variable ELO, so that our model learns to play against a potential human user. In fact, in a sense it is easier to learn to beat Stockfish because of it's predictability, due to the constant performance of the best moves.
