from algorithm_utils_br import *

def game(Q, num_turns = 20, sleep = 0):
    game_result = ""                    #win or loss 
  
    #Initialise the current state with the starting state
    cstate = np.array([[2,4,5],[6,4,2]])                   #must be the same the model has been trained on

    #First move of the game (white)
    #cmove = "pfwd"                                         #deterministic
    cmove = greedy(array_to_int(cstate), Q)[1]           #epsilon-greedy
    if cmove in ["pfwd", "ppfwd"]:                          #Retrieve the piece moved
        p = "P" 
    else: 
        p = "Kw" 

    #Internal Loop: Turns per game----------------------------------------------------------------------------
    for turn in range(num_turns): 
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
        
        #Black turn: USER---------------------------------------------------------
        try:
            bcmove = input()
            black_action = stock_ita(bcmove) #Translate the black move from standard notation to our notation
        except:
            print("Invalid move, try again:")
            bcmove = input()
            black_action = stock_ita(bcmove) #Translate the black move from standard notation to our notation
            break
        s_prime_after_black = black_move(s_prime_before_black, black_action) #Update the location matrix with the black move
        
        #Pass the black move to the chess engine and show it on chessboard
        board.push(chess.Move.from_uci(bcmove))
        display(board)
        time.sleep(sleep)
        clear_output(wait=True)
        
        if (s_prime_before_black[:,0] == s_prime_after_black[:,2]).all():
            game_result = "win"
            break
        elif s_prime_after_black[0][0] == 7:
            game_result = "loss"
            break
        
        #-------------------------------------------------------------------------

        #trasform arrays into integers 
        int_s_prime_after_black = array_to_int(s_prime_after_black)

        #Choose the next white action and move (as written in Q and in our notation)
        try:
            _, move_prime = greedy(int_s_prime_after_black, Q)
        except: 
            if s_prime_after_black[0][0] == 7:
                game_result = "loss"  
                print("legal_move() exception:  \n YOU LOSE")
                display(chess.Board(init_game(s_prime_after_black, queen = True)))
                time.sleep(2)
                clear_output(wait=True)  
            else:   
                game_result = "win"
                print("legal_move() exception:  \n STALEMATE")
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
    
    return game_result

#------------------------------------

def greedy(s_check, Q):
    '''
    Follow greedy policy.
    - Input: current state in integer form (array_to_int(location_matrix)), Q.
    - Output: action_in_notation: move in our notation, state_prime_pair: move in the form of pair of states.
    '''
    state_prime_pair = random.choice([a for a,v in Q[s_check].items() if v == max([(v,a)[0] for a, v in Q[s_check].items()])])
    L = int_to_array(state_prime_pair[0], small = True)
    L_prime = int_to_array(state_prime_pair[1], small = True)
    action_in_notation = red_mat_move(L, L_prime)
    
    return state_prime_pair, action_in_notation