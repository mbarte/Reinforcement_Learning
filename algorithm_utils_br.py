import random
import chess
import time
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import pandas as pd

from dynamics_br import *

#-----------------------------------------------------------------------------
#---------------ALGORITHM UTILS-----------------------------------------------
#-----------------------------------------------------------------------------

def count_visit(visited_state, N):
    '''
    count the number of visits for each of the possible states visited.
    - Input: the current visited state and the N.
    - Output: Updated N.
    '''
    visited_state = array_to_int(reduce_array(visited_state), small = True)
    N[visited_state] += 1
    return N

#------------------------------------------------------------

def epsilon(s_check, N):
    '''
    - Input: the current state and N.
    - Output: 1 over the number of vistis of the current state.
    '''
    return 1/N[array_to_int(reduce_array(int_to_array(s_check)), small = True)]

#------------------------------------------------------------

def epsilon_greedy(s_check, Q, N):
    '''
    Choose whether to be explorative or greedy in the next white move.
    - Input: current state in integer form (array_to_int(location_matrix)), Q, N.
    - Output: action_in_notation: move in our notation, state_prime_pair: move in the form of pair of states.
    '''
    x = np.random.uniform()
    if x < epsilon(s_check, N):
        possible_moves = sorted(list(legal_move(int_to_array(s_check))), reverse = True)
        action_in_notation = random.choices(possible_moves, weights = weights(possible_moves))[0]
        
        L = array_to_int(reduce_array(int_to_array(s_check)), small = True)
        L_prime = array_to_int(reduce_array(move(int_to_array(s_check), action_in_notation)[0]), small = True)
        state_prime_pair = (L, L_prime)
        
    else:
        state_prime_pair = random.choice([a for a,v in Q[s_check].items() if v == max([(v,a)[0] for a, v in Q[s_check].items()])])
        L = int_to_array(state_prime_pair[0], small = True)
        L_prime = int_to_array(state_prime_pair[1], small = True)
        action_in_notation = red_mat_move(L, L_prime)
    
    return state_prime_pair, action_in_notation

#------------------------------------------------------------

def states(): 
    '''
    Define all the possible states. (8*8*8*6*5 = 15360 states stored)
    - Input: --
    - Output: A list with all the possible states.
    In order to generalise to more states, change ranges accordingly.
    '''
    N = list()
    pos = 0
    for a in range(2, 8):       #pawn starts in g3
        for b in range(8):  
            for c in range(8):
                for d in range(3,8): #white king limited in columns d,e,f,g,h
                    for e in range(8):
                        N.append(np.array([[a,b,c],[6,d,e]]))
                        pos += 1
    return N

#------------------------------------------------------------

def reduce_array(array):
    '''
    Retrieve the location matrix for white pieces only (e.g. array([[1,0,7],[3,3,3]])  --> array([[1,0],[3,3]]).
    - Input: Location matrix in array form.
    - Output: Reduced location matrix in array form.
    '''
    reduced_array = np.zeros((2,2), dtype = 'int')
    reduced_array[0][0] = array[0][0]
    reduced_array[0][1] = array[0][1]
    reduced_array[1][0] = array[1][0]
    reduced_array[1][1] = array[1][1]
    return reduced_array

#------------------------------------------------------------

def red_mat_move(L, L_prime):
    '''
    Retrieve the white move from a pair of reduced location matrices.
    - Input: Reduced current state, reduced next state.
    - Output: white move in our notation (e.g. 'fwd').
    '''
    if L[0][0] != L_prime[0][0]: 
        mossa = (L_prime - L) @ [1,0]
        my_list = list(p_moves.values())
        my_keys = list(p_moves.keys())
    else: 
        mossa = (L_prime - L) @ [0,1]
        my_list = list(k_moves.values())
        my_keys = list(k_moves.keys())
        
    for i in range(0, len(my_list)):
        if (np.array_equal(mossa, my_list[i])): 
            mossa = my_keys[i]
            
    return mossa

#------------------------------------------------------------

def weights(possible_moves):
    '''
    Compute weights to balance the probability of moving white pawn or white king. 
    - Input: list of possible moves.
    - Output: list of corresponding weights for each move.
    '''
    weights = []
    if "pfwd" in possible_moves and "ppfwd" in possible_moves:
        weights = [0.25, 0.25]
        [weights.append(round(1/(2*(len(possible_moves)-2)), 4)) for x in range(len(possible_moves)-2)]
    elif "pfwd" in possible_moves:
        weights = [0.5]
        [weights.append(round(1/(2*(len(possible_moves)-1)), 4)) for x in range(len(possible_moves)-1)]
    else:
        [weights.append(round(1/(len(possible_moves)), 4)) for x in range(len(possible_moves))]
    return weights

        
#------------------------------------------------------------

def trend_plot(results):
    '''
    Plots the results.
    '''
    vals = results #trend list
    
    df = pd.DataFrame(data=vals, columns=['Value'])
    df['difference'] = df.diff()
    df['condition'] = (df.difference > 0).astype(int)
    df['group'] = df.condition.diff().abs().cumsum().fillna(0).astype(int) + 1
    
    fig, ax = plt.subplots()
    # fail safe only
    ax.plot(df.Value, color='blue')
    
    # decides if starts in descend
    # (first difference is NaN therefore first condition 0 no matter what)
    red = df.condition.iloc[1] == 1
    last = pd.DataFrame()
    for i in range(df.group.max() + 1):
        group = pd.concat([last, df.Value[df.group == i]])
        last = group.iloc[-1:]
        red = not red
    
        ax.plot(group, color='green' if red else 'red')
        
#------------------------------------------------------------
