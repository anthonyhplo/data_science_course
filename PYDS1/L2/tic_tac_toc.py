import numpy as np

def game_update(game_board, player):
    input_list = []
    while not len(input_list) == 2 or not input_list[0].isdigit() or not input_list[1].isdigit() or not 0 <= int(input_list[0]) < game_board.shape[0] or not 0 <= int(input_list[1]) < game_board.shape[0] or not game_board[int(input_list[0]),int(input_list[1])] == 0:
        user_input = input('Player{}, Please enter your row,col position, e.g.:0,0):'.format(str(player)))
        input_list = user_input.split(",")
    row_input = int(input_list[0])
    col_input = int(input_list[1])
    game_board[row_input,col_input] = player
    return game_board, (row_input, col_input)

def win_check(game_board, player, update_step, board_size, left_diag, right_diag):
    #check horizontal
    if len(set(game_board[update_step[0],])) == 1 and list(set(game_board[update_step[0],]))[0] == player:
        return player
    #Check vertical
    elif len(set(game_board[:,update_step[1]])) == 1 and list(set(game_board[:,update_step[1]]))[0] == player:
        return player
    #cross check: left to right
    elif update_step in left_diag and len(set([game_board[i[0],i[1]] for i in left_diag])) == 1:
        return player
    elif update_step in right_diag and len(set([game_board[i[0],i[1]] for i in right_diag])) == 1:
        return player
    else:
        return 0

def tic_tac_toe(board_size):
    # initial State
    player_1 = 1
    player_2 = 2
    game_board = np.zeros((board_size, board_size))
    max_step = board_size*board_size
    step = 0
    diagonal_left = [(i,i) for i in range(board_size)]
    diagonal_right = [(i, board_size-1-i) for i in range(board_size)]
    print(game_board)
    continue_play = True
    while continue_play:
        for player in [1,2]:
            game_board, update_step = game_update(game_board, player)
            print(game_board)
            game_status = win_check(game_board, player, update_step, board_size, diagonal_left, diagonal_right)
            step += 1
            if game_status != 0:
                print("Game ended, player{} wins!".format(str(game_status)))
                continue_play = False
                break
            if step == max_step:
                print("Draw game")
                continue_play = False
                break

tic_tac_toe(3)
            