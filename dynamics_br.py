import numpy as np
from stockfish import Stockfish
import io

#examples of L_starts (Location matrix)
#L = np.array([[1,0,7],[3,3,3]])    
#L = np.array([[1,0,6],[3,3,3]])
#L = np.array([[1,2,4],[6,4,0]])
# L = np.array([[2,3,5],[6,4,2]])   
#For different location matrices change states() accordingly

#Selection matrix
S = {"P": np.array([1,0,0]), 
     "Kw": np.array([0,1,0]), 
     "Kb": np.array([0,0,1])
     }
    
p_moves = {"pfwd": np.array([1,0]),"ppfwd": np.array([2,0])}

k_moves = {"kfwd": np.array([1,0]),
            "kbwd": np.array([-1,0]),
            "kright": np.array([0,1]),
            "kleft": np.array([0,-1]),
            "kfr": np.array([1,1]),
            "kfl": np.array([1,-1]),
            "kbr": np.array([-1,1]),
            "kbl": np.array([-1,-1])
        }
#It will eventually learn not to do the commented actions in a sufficient number of games, 
# but in order to reduce the number of possible states, we simply prevent it from doing them.

kb_moves = {"kfwd": np.array([1,0]),
            "kbwd": np.array([-1,0]),
            "kright": np.array([0,1]),
            "kleft": np.array([0,-1]),
            "kfr": np.array([1,1]),
            "kfl": np.array([1,-1]),
            "kbr": np.array([-1,1]),
            "kbl": np.array([-1,-1])
        } 

#stoc = Stockfish(path="C:/Users/pc/Desktop/UNIMI- DSE/First year/Reinforcement Learning/Project/stockfish_15.1_win_x64_popcnt/stockfish-windows-2022-x86-64-modern")             
#stoc = Stockfish(path="C:/Users/Guido/Desktop/RL/Project/stockfish_15.1_win_x64_popcnt/stockfish-windows-2022-x86-64-modern")

#---------------------------------------------
#-----------DYNAMICS FUNCTIONS----------------
#---Regulating a single agent (white player)--
#---------------------------------------------


def init_game(L, queen = False):
#Inspired by: https://stackoverflow.com/questions/56754543/generate-chess-board-diagram-from-an-array-of-positions-in-python
    '''
    Visualise the current state (positions on the chessboard).
    - Input: Location matrix. If queen = True it generates the fen with the pawn promoted to queen.
    - Output: Corresponding fen notation.
    '''
    fen_matrix = [["em" for c in range(8)] for r in range(8)]
    if queen:
      fen_matrix[L[0][0]][L[1][0]] = "wq"
    else:
      fen_matrix[L[0][0]][L[1][0]] = "wp"
    fen_matrix[L[0][1]][L[1][1]] = "wk"
    fen_matrix[L[0][2]][L[1][2]] = "bk"

    # Use StringIO to build string more efficiently than concatenating
    with io.StringIO() as s:
        for row in range(len(fen_matrix)-1,-1,-1):
            empty = 0
            for cell in range(0,len(fen_matrix)):
                c = fen_matrix[row][cell]
                if c[0] in ('w', 'b'):
                    if empty > 0:
                        s.write(str(empty))
                        empty = 0
                    s.write(c[1].upper() if c[0] == 'w' else c[1].lower())
                else:
                    empty += 1
            if empty > 0:
                s.write(str(empty))
            s.write('/')
        # Move one position back to overwrite last '/'
        s.seek(s.tell() - 1)
        return s.getvalue()[:-1]   
    
#---------------------------------------------

def legal_move(L, limit = True):
    '''
    Retrieve the possible moves from the current state.
    - Input: Location matrix. If limit = False the white king can move without restrictions on any column.
    - Output: Set of possible moves for the white player from the current state.
    '''
    lm = set()           #Set of possible moves
    if  L[0][0] == 1 \
        and not np.array_equiv((L[:,1] - L[:,0]), np.array([2,0])) and not np.array_equiv((L[:,1] - L[:,0]), np.array([1,0])) \
        and not np.array_equiv((L[:,2] - L[:,0]), np.array([2,0])) and not np.array_equiv((L[:,2] - L[:,0]), np.array([1,0])):
      lm.add("pfwd")
      lm.add("ppfwd")
    elif not L[0][0] == 7 \
        and not np.array_equiv((L[:,1] - L[:,0]), np.array([1,0])) \
        and not np.array_equiv((L[:,2] - L[:,0]), np.array([1,0])):
      lm.add("pfwd")
        
    my_list = list(k_moves.values())
    my_keys = list(k_moves.keys())
    
    obstacles = np.zeros(shape = (8,8), dtype = int)  #matrix containing occupied squares
    candidates = np.zeros(shape = (8,8), dtype = int) #matrix containing possible squares for the white king
    diff = list()                                      
    poss_squares = list()                             #list containing the possible squares coordinates for white king

    #Extract the pieces coordinates to fill the 'obstacles' and 'candidates' matrices
    px, py = L[0][0], L[1][0]
    kbx, kby = L[0][2], L[1][2]
    kwx, kwy = L[0][1], L[1][1]
    kw_coord = [kwx, kwy]


    #Fill the matrices
    obstacles[px][py] = 1
    obstacles[kwx][kwy] = 1

    for i in range(max(kbx-1,0), min(8,kbx+2)):
      for j in range(max(kby-1,0), min(8,kby+2)): 
       obstacles[i][j] = 1

    for i in range(max(kwx-1,0), min(8,kwx+2)):
      for j in range(max(kwy-1,0), min(8,kwy+2)): 
        candidates[i][j] = 10
        
    #Combine the matrices    
    config = candidates + obstacles
    #Limit the columns of the white king (not allowed: a,b,c)
    if limit == True:
      for j in (0,1,2): 
        for i in range(8):
          config[i][j] += 1
          
    #Extraxt the coordinates
    for i in range(0, 8): 
      for j in range(0,8):
       if(config[i][j]== 10): 
          poss_squares.append([i,j])

    for k in range(0, len(poss_squares)):   
      diff.append(np.array(poss_squares[k]) - np.array(kw_coord))
  
    for i in range(0, len(my_list)):
      for j in range(0, len(diff)):
        if (diff[j] == my_list[i]).all(): 
          lm.add(my_keys[i])
    
    return lm

#---------------------------------------------

def legal_state(L):
    '''
    Retrieve the possible future states from the current one.
    - Input: Location matrix.
    - Output: List of possible future states.
    '''
    ls = list()
    L_primes = list()
    a = legal_move(L)
    for move in a:
      if move in ("pfwd", "ppfwd"):
        ls.append(L @ np.atleast_2d(S["P"]).T + np.atleast_2d(p_moves[move]).T)
        L_primes.append(L + np.atleast_2d(p_moves[move]).T @ np.atleast_2d(S["P"]))
      else: 
        ls.append(L @ np.atleast_2d(S["Kw"]).T + np.atleast_2d(k_moves[move]).T)
        L_primes.append(L + np.atleast_2d(k_moves[move]).T @ np.atleast_2d(S["Kw"]))

    return L_primes  

#---------------------------------------------

def move(L, action):
    '''
    Update the current state to reach the future one thanks to a specific white move.
    - Input: Location matrix and white move in our notation.
    - Output: Future state, move in Stockfish notation and piece moved.
    '''
    piece = ""
    if action in ("pfwd", "ppfwd"):
      L_prime = np.zeros(shape = (2,3))
      L_prime = L + np.atleast_2d(p_moves[action]).T @ np.atleast_2d(S["P"])   #atleast_2d necessary to perform matix multiplication
      piece = "P"    #white moved pawn
    else: 
      L_prime = np.zeros(shape = (2,3))
      L_prime = L + np.atleast_2d(k_moves[action]).T @ np.atleast_2d(S["Kw"]) 
      piece = "Kw"   #white moved king
   
    choosen_move = ita_stock(L, L_prime, piece)
    return L_prime, choosen_move, piece

#---------------------------------------------
#
#Not useful in the final version
#
#def reward(L, L_prime):
#  '''
#  Retrieve the reward associated to the white move.
#  - Input: Current location matrix, future state (after white move)
#  - Output: Reward (float) associated to the move.
#  '''
#  r = 0
#  if np.array_equiv(L[:,0], L_prime[:,0]):  #white player moved the king
#    piece = "K"
#    if abs(7 - L[0][0]) < abs(L[1][2] - L[1][0]):
#      r = 0
#    elif abs(7 - L[0][0]) >= abs(L[1][2] - L[1][0]):  
#      if L_prime[1][1] - L[1][0] < 2 :   ##re nella colonna vicino al pedone
#        r = 5
#      else:
#        r = 2
#  else:                                   #white player moved the pawn
#    piece = "P"
#    if L_prime[0][0] == 7:
#      r = 200
#    else:
#      if L_prime[0][0] == L[0][1]:  #pedone stessa riga di re
#        r = -5
#      if abs(7 - L[0][0]) >= abs(L[1][2] - L[1][0]):
#        r = r+0
#        if dist(L_prime[:,1], L_prime[:,0]) <= np.sqrt(2): 
#          r = r+3
#        else: 
#          r = r+1 
#      elif abs(7 - L[0][0]) < abs(L[1][2] - L[1][0]):
#        r = r+15
#  return piece, r   
#
#---------------------------------------------
#
#Not useful in the final version
#
#def successors(L):
#    '''
#    For all the possible states from the current one, yields a list of possible moves with the associated reward.
#    - Input: Current location matrix.
#    - Output: [[piece1, move1, r1], [piece2, move2, reward2], ...], where move is in stockfish notation.
#    '''
#    data = []
#    legal_moves = legal_move(L)
#    for a in legal_moves:
#      L_prime = move(L, a)[0] 
#      piece = move(L,a)[2]
#      s_prime = ita_stock(L, L_prime, piece)   #return the action in stockfish notation (Ex:'d2d3')
#      r = reward(L, L_prime)[1]
#      data.append([piece, s_prime, r])   
#    
#    return data 
#
#---------------------------------------------

def black_move(L, black_action):  
    '''
    Update the current state to reach the future one thanks to a specific black move.
    - Input: Location matrix and black move in our notation.
    - Output: Future state.
    '''
    L_prime = np.zeros(shape = (2,3))
    L_prime = L + np.atleast_2d(kb_moves[black_action]).T @ np.atleast_2d(S["Kb"])  

    return L_prime

#---------------------------------------------
#---------COMMUNICATION FUNCTIONS-------------
#---------------------------------------------

def stock_ita(s): 
    '''
    Translate the black move from stockfish notation to our notation.
    - Input: string in the form 'd2d3'.
    - Output: string in the form 'fwd'.
    '''
    stockita = {"a": 0, "b": 1, "c":2, "d":3, "e": 4, "f": 5, "g": 6, "h": 7 }
    a,b,c,d = list(s)    #d 2 d 3
    kby_move = stockita[c] - stockita[a]   #black king movement along y axis
    kbx_move = int(d) - int(b)    #black king movement along x axis
    #our indexes are from 0 to 7, so we must scale his move that ranges from 1 to 8
    kb_array = np.array([kbx_move, kby_move]) 
    blk_move_in_notation = [i for i in kb_moves.keys() if (kb_moves[i] == kb_array).all()]
    
    return blk_move_in_notation[0]
  
#---------------------------------------------

def ita_stock(L,L_prime, piece):
    '''
    retrieve the move in the stockfish notation (e.g. 'd2d3').
    - Input: Current location matrix, future location matrix, moved piece
    - Output: string in stockfish notation (e.g. 'd2d3').
    '''
    itastock = {0: "a", 1: "b", 2:"c", 3:"d", 4: "e", 5: "f", 6: "g", 7: "h" }
    if piece == "P":
        a,b = L[0][0]+1, L[1][0]   #add 1 to the row to match the range.
        a_prime,b_prime = L_prime[0][0]+1 ,L_prime[1][0]
    else: 
        a,b = L[0][1]+1, L[1][1]
        a_prime,b_prime = L_prime[0][1]+1, L_prime[1][1]
    py = itastock[b]
    px = str(a)
    py_prime = itastock[b_prime]
    px_prime = str(a_prime)
    return py + px + py_prime + px_prime

#---------------------------------------------

def array_to_int(L, small = False): 
  '''
  trasform an array in an integer by row (e.g: array([[1,0,7],[3,3,3]])  --> 107333).
  - Input: Location matrix in array form.
  - Output: An integer representing the location matrix.
  '''
  number = 0  
  
  if small == True:
    number = number + 1e3*L[0][0]
    number = number + 1e2*L[0][1]
    number = number + 1e1*L[1][0]
    number = number + 1e0*L[1][1]
    
  else: 
    number = number + 1e5*L[0][0]
    number = number + 1e4*L[0][1]
    number = number + 1e3*L[0][2]
    number = number + 1e2*L[1][0]
    number = number + 1e1*L[1][1]
    number = number + 1e0*L[1][2]
    
  return int(number)

#---------------------------------------------

def int_to_array(number, small = False): 
  '''
  Trasform an integer in an array (i.e: 107333 --> array([[1,0,7],[3,3,3]]).
  - Input: an integer.
  - Output: array representing the Location matrix.
  '''
  if not small:
    arr = np.zeros((2,3), dtype = 'int')
    l = [int(x) for x in str(number)]
    arr[0][0] = l[0]
    arr[0][1] = l[1]
    arr[0][2] = l[2]
    arr[1][0] = l[3]
    arr[1][1] = l[4]
    arr[1][2] = l[5]
    
  else: 
    arr = np.zeros((2,2), dtype = 'int')
    l = [int(x) for x in str(number)]
    arr[0][0] = l[0]
    arr[0][1] = l[1]
    arr[1][0] = l[2]
    arr[1][1] = l[3]

  return arr

#---------------------------------------------