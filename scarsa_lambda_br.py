from collections import defaultdict
from algorithm_utils_br import *

def scarsa_lambda(L_start, gamma_ = 1, lambda_ = 0.8, eta = 0.01, num_games = 1000, num_turns = 50, sleep = 0):
    '''
    - input: initial location matrix, gamma_, lambda_, eta (learning rate), num_games = number of games to train the model, num_turns = maximum number of tunrs per game, 
    sleep = seconds before clearing output and displaying chessboard after turn
    '''
    #---------------------------------------------
    game_result = ""                    #win or loss 
    trend_results = []   #keep trace of the winning trend and number of moves to win
    trend_results.append(0)

    reward_value = 0
    #---------------------------------------------
    #---------------------------------------------
    #Define and initialise Q and E: 
    # --> keys: "hashed" possible states
    # --> values: possible actions (pairs of "hashed" states) and reward for performing such transitions
    Q = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    E = defaultdict(lambda: defaultdict(lambda: float("-inf")))
    for state in states():
        prov = legal_state(state)
        s = array_to_int(state)
        for i in prov:
            a = (array_to_int(reduce_array(int_to_array(s)), small = True), array_to_int(reduce_array(i), small = True))
            Q[s][a] = 0
            E[s][a] = 0

    #Define N: The counter for visits to the reduced states (see reduce_array())
    N = dict([(array_to_int(reduce_array(s), small = True), 1) for s in states()])

    #stoc = Stockfish(path="C:/Users/Guido/Desktop/RL/Project/stockfish_15.1_win_x64_popcnt/stockfish-windows-2022-x86-64-modern")
    stoc = Stockfish(path="C:/Users/pc/Desktop/UNIMI- DSE/First year/Reinforcement Learning/Project/stockfish_15.1_win_x64_popcnt/stockfish-windows-2022-x86-64-modern")
    #---------------------------------------------
    #---------------------------------------------

    #Start the external loop: Iterate over the number of games (Episodes)---------------------------------------------
    for episode in range(num_games):
        
        moves_per_episode = list()
        states_per_episode = list()

        #print("Game number: ", episode + 1)
        
        #Initialise the current state with the starting state
        cstate = L_start
        
        #First move of the game (white)
        #cmove = "kfr"                                          #deterministic
        cmove = epsilon_greedy(array_to_int(cstate), Q, N)[1]   #epsilon-greedy
        if cmove in ["pfwd", "ppfwd"]:                          #Retrieve the piece moved
            p = "P" 
        else: 
            p = "Kw" 

        #Internal Loop: Turns per game----------------------------------------------------------------------------
        for turn in range(num_turns): 
            
            states_per_episode.append(array_to_int(cstate))
            
            #Initialise the game in the first turn
            if turn == 0: 
                board = chess.Board(init_game(cstate))  #Initialise the board which dialogue with the chess engine
                display(board)                          #visualise the chessboard
                time.sleep(sleep)
                clear_output(wait=True)  
                
                #Make the first white move:
                s_prime_before_black = move(cstate, cmove)[0]               #Update the location matrix after white move ("pfwd")
                white_action = ita_stock(cstate, s_prime_before_black, p)   #translate the white move in standard notation ("d2d3")

            #Pass the white move to the chess engine and show it on chessboard
            board.push(chess.Move.from_uci(white_action))
            display(board)
            time.sleep(sleep)
            clear_output(wait=True)
            
            #Black turn---------------------------------------------------------------
            try:
                stoc.set_fen_position(board.fen())
                bcmove = stoc.get_best_move()
                black_action = stock_ita(bcmove)  #Translate the black move from standard notation to our notation
            except:
                game_result = "loss"
                print("STALEMATE")
                break
            s_prime_after_black = black_move(s_prime_before_black, black_action) #Update the location matrix with the black move
            
            #Pass the black move to the chess engine and show it on chessboard
            board.push(chess.Move.from_uci(bcmove))
            display(board)
            time.sleep(sleep)
            clear_output(wait=True)
            
            #Check
            if (s_prime_before_black[:,0] == s_prime_after_black[:,2]).all() or turn == num_turns -1:
                game_result = "loss"
                trend_results.append(trend_results[-1]-1)
                break
            elif s_prime_after_black[0][0] == 7:
                game_result = "win"
                trend_results.append(trend_results[-1]+1)
                display(chess.Board(init_game(s_prime_after_black, queen = True)))
                time.sleep(sleep)
                clear_output(wait=True) 
                break
            #-------------------------------------------------------------------------
        
            # #Update count of visits to reduced states
            N = count_visit(s_prime_after_black, N)
            
            #Update R
            state_pair = (array_to_int(reduce_array(cstate), small = True), 
                        array_to_int(reduce_array(s_prime_before_black), small = True)) #Retrieve the white action as a pair, as stored in R
            
            moves_per_episode.append(state_pair)

            #trasform array into integer
            int_s_prime_after_black = array_to_int(s_prime_after_black)  

            #Choose the next white action and move (as written in Q and in our notation)
            try:
                _, move_prime = epsilon_greedy(int_s_prime_after_black, Q, N)
            except: 
                if s_prime_after_black[0][0] == 7:
                    game_result = "win"
                    trend_results.append(trend_results[-1]+1)
                    print("legal_move() exception:  \n WIN --> skipping to the next game")
                    display(chess.Board(init_game(s_prime_after_black, queen = True)))
                    time.sleep(sleep)
                    clear_output(wait=True)  
                else:   
                    print("legal_move() exception:  \n STALEMATE --> skipping to the next game")
                    game_result = "loss"
                    trend_results.append(trend_results[-1]-1)
                break
                
            if move_prime in ["pfwd", "ppfwd"]: #Retrieve the piece moved
                p = "P"
            else: 
                p = "Kw"
            #-------------------------------------------------------------------------
                
            #Overwrite states to begin the new turn
            cstate = s_prime_after_black     #Update the current state  
            cmove = move_prime               #Update the current white move(with the one chosen by epsilon-greedy Algorithm at the previous step)
            
            #White Move for the next-iteration turn-----------------------------------
            s_prime_before_black = move(cstate, cmove)[0]                   #Update the location matrix with the white move
            white_action = ita_stock(cstate, s_prime_before_black, p)       #translate the white move in standard notation (i.e "d2d3")
            #-------------------------------------------------------------------------

        #Assign the reward: positive reward for moves that led to a victory (proportional to their number)
        if game_result == 'win':
            reward_value = (num_turns*10)/(len(moves_per_episode)+1)
        else: 
            reward_value = -10    #negative  otherwise
        game_result = ""
        
        #Algorithm core-----------------------------------------------------------
        for i in range(len(moves_per_episode)-1):
            int_s_prime = states_per_episode[i+1]
            state_pair_prime = moves_per_episode[i+1]
            int_s = states_per_episode[i]
            state_pair = moves_per_episode[i]

            #Compute delta
            delta = reward_value + gamma_ * Q[int_s_prime][state_pair_prime] - Q[int_s][state_pair]     
            E[int_s][state_pair] = E[int_s][state_pair] + 1                                            
            
            #Update Q and E
            Q[int_s][state_pair] = Q[int_s][state_pair] + eta * delta * E[int_s][state_pair] 
            E[int_s][state_pair] = gamma_ * lambda_ * E[int_s][state_pair]  
    
    return Q, N, E, trend_results