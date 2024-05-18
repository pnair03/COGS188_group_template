import random
import heapq
import pygame
import sys
import numpy as np

#Constants
BOARD_SIZE = (28, 31)
GRID_SIZE = 20
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 217, 0)
BLUE = (33,34,222)
PELLET = (255, 184, 151)
BLINKY = (240, 0, 2)
PINKY = (249, 154, 205)
INKY = (5, 205, 239)
CLYDE = (255, 140, 2)

#0: Pellet Space, 1: Wall, 2: Boundary not accesible by anyone, 3: No Pellet space
maze_layout = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 1, 2, 2, 2, 2, 2, 2, 1, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 3, 1, 2, 2, 2, 2, 2, 2, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
#Ghost Behavior Modes
mode_schedule = [
    ('scatter', 7),
    ('chase', 20),
    ('scatter', 5),
    ('chase', 20)
    ('scatter', 5),
    ('chase', float('inf'))    
]

#Starting Positions
lives = 3
score = 0
pacman_position = {'position':random.choice([[14, 22], [13,22]]), 'direction':'up'}
ghost_positions = {'blinky':[15, 12], 'pinky':[15,12], 'inky':[15,12], 'clyde':[14,10]}
power_pellet_positions = [(1, 23), (1, 3), (26, 3), (26, 23)]

#Action Space
action_space = ['up', 'down', 'left', 'right', 'stay']

#Initialize Gameboard States with state_type and has_pellet info
gameboard_state = {}
for row in range(len(maze_layout)):
    for col in range(len(maze_layout[row])):
        state_type = maze_layout[row][col]
        gameboard_state[(col, row)] = {'state': state_type, 'has_pellet': state_type == 0}

# Find total number of pellets
total_pellets = sum([1 for state in gameboard_state.values() if state['state']==0])

# Initialize Pygame
pygame.init()   

# Set up the screen
screen = pygame.display.set_mode((BOARD_SIZE[0] * GRID_SIZE, BOARD_SIZE[1] * GRID_SIZE))
pygame.display.set_caption('Pacman')

def reset():
    global pacman_position
    global ghost_positions
    global lives
    
    lives -=1
    if lives < 1:
        print("GAME OVER")
        pygame.quit()
        sys.exit()
    else:
        ghost_positions = {'blinky':[15,12], 'pinky':[15,12], 'inky':[15,12], 'clyde':[14,10]}
        pacman_position = {'position':random.choice([[14, 22], [13,22]]), 'direction':'up'}


def draw_board():
    """
    Draws the board with the correct visual representation for each state on the board

    Parameters:
    None

    Returns:
    None
    """
    global score
    screen.fill(BLACK)

    #Display Score "text"
    font = pygame.font.Font('freesansbold.ttf', 20)
    text = font.render(f"Score: {score}", True, WHITE, BLACK)
    textRect = text.get_rect()
    textRect.center = (2*GRID_SIZE,BOARD_SIZE[1]*GRID_SIZE-10)
    screen.blit(text, textRect)

    #Draw out map and its boundaries
    for row in range(len(maze_layout)):
        for col in range(len(maze_layout[row])):
            cell = gameboard_state[(col, row)]
            state_type = cell['state']
            x = col * GRID_SIZE
            y = row * GRID_SIZE

            if state_type == 1:
                pygame.draw.rect(screen, BLUE, (x, y, GRID_SIZE, GRID_SIZE), 1)
            elif state_type == 0 and cell['has_pellet']:
                pygame.draw.circle(screen, PELLET, (x + GRID_SIZE // 2, y + GRID_SIZE // 2), 3)

    # Mark Pacman Position
    x_pacman = pacman_position['position'][0] * GRID_SIZE + GRID_SIZE // 2
    y_pacman = pacman_position['position'][1] * GRID_SIZE + GRID_SIZE // 2
    pygame.draw.circle(screen, YELLOW, (x_pacman, y_pacman), 10)

    # Mark Ghost(s) Position(s)
    #Blinky
    x_blinky = ghost_positions['blinky'][0] * GRID_SIZE + GRID_SIZE // 2
    y_blinky = ghost_positions['blinky'][1] * GRID_SIZE + GRID_SIZE // 2
    pygame.draw.circle(screen, BLINKY, (x_blinky, y_blinky), 10)

    #Pinky
    x_pinky = ghost_positions['pinky'][0] * GRID_SIZE + GRID_SIZE // 2
    y_pinky = ghost_positions['pinky'][1] * GRID_SIZE + GRID_SIZE // 2
    pygame.draw.circle(screen, PINKY, (x_pinky, y_pinky), 10)

    #Inky
    x_inky = ghost_positions['inky'][0] * GRID_SIZE + GRID_SIZE // 2
    y_inky = ghost_positions['inky'][1] * GRID_SIZE + GRID_SIZE // 2
    pygame.draw.circle(screen, INKY, (x_inky, y_inky), 10)

    #Clyde
    x_clyde = ghost_positions['clyde'][0] * GRID_SIZE + GRID_SIZE // 2
    y_clyde = ghost_positions['clyde'][1] * GRID_SIZE + GRID_SIZE // 2
    pygame.draw.circle(screen, CLYDE, (x_clyde, y_clyde), 10)


def move_pacman(direction):
    """
   Checks if Pacman can move in direction specified, if can updates Pacman's new position

    Parameters:
    direction (str): A string representing which direction Pacman wishes to move.

    Returns:
    None
    """
    global pacman_position
    global score
    global total_pellets
    x, y = pacman_position['position']

    if direction == 'up' and maze_layout[y - 1][x] != 1:
        y -= 1
        pacman_position['direction'] = 'up'

    elif direction == 'down' and maze_layout[y + 1][x] != 1:
        y += 1
        pacman_position['direction'] = 'down'

    elif direction == 'left':
        if x == 0 and y == 13:
            x = 27
            pacman_position['direction'] = 'left'
        elif maze_layout[y][x - 1] != 1:
            x -= 1
            pacman_position['direction'] = 'left'

    elif direction == 'right':
        if x == 27 and y == 13:
            x = 0
            pacman_position['direction'] = 'right'
        elif maze_layout[y][x + 1] != 1:
            x += 1
            pacman_position['direction'] = 'right'


    pacman_position['position'] = [x, y]
    if gameboard_state[(x, y)]['has_pellet']:
        gameboard_state[(x, y)]['has_pellet'] = False
        total_pellets -= 1
        score += 10
        check_win()

def check_win():
    global score
    global total_pellets
    if total_pellets == 0:
        print(f"Congratulations, you won!! You have a total score of {score}!")
        pygame.quit()
        sys.exit()


# A* Algorithm to move Ghosts
# Heuristic function for A* (Manhattan distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* Algorithm Implementation
def astar(gameboard_state, start, goal):
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))
    
    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)
            return path[::-1]

        close_set.add(current)

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            if (neighbor in close_set or neighbor not in gameboard_state) or (gameboard_state[neighbor]['state'] in [1, 2]):
                continue

            potential_score = gscore[current] + 1

            if neighbor in gscore and potential_score >= gscore[neighbor]:
                continue

            came_from[neighbor] = current
            gscore[neighbor] = potential_score
            fscore[neighbor] = gscore[neighbor] + heuristic(neighbor, goal)
            heapq.heappush(oheap, (fscore[neighbor], neighbor))
    
    return [] 

# Adjust positions for A* function call
def move_ghosts():
    global pacman_position
    pacman_position['position'] = tuple(pacman_position['position'])
    #Blinky
    blinky_position = tuple(ghost_positions['blinky'])
    path = astar(gameboard_state, blinky_position, pacman_position['position'])
    if path and len(path) > 1:
        ghost_positions['blinky'] = list(path[1])

    #Pinky
    #Finds new Pinky Target for 'goal' parameter in Astar function
    pinky_target = pacman_position['position']
    if pacman_position['direction'] == 'up':
        pinky_target = [pinky_target[0]-4, pinky_target[1]-4]
    elif pacman_position['direction'] == 'down':
        pinky_target = [pinky_target[0], pinky_target[1]+4]
    elif pacman_position['direction'] == 'left':
        pinky_target = [pinky_target[0]-4, pinky_target[1]]
    else:
        pinky_target = [pinky_target[0]+4, pinky_target[1]]

    #Adjusts if target is out of boundaries
    pinky_target[0] = max(1, min(pinky_target[0], BOARD_SIZE[0]-2))
    pinky_target[1] = max(1, min(pinky_target[1], BOARD_SIZE[1]-3))

    #Adjsuts if target is a wall or unmovable space
    while gameboard_state[(pinky_target[0], pinky_target[1])] in [0,3]:
        if pacman_position['direction']['up']:
            pinky_target[1] -=1
        elif pacman_position['direction']['down']:
            pinky_target[1] += 1
        elif pacman_position['direction']['left']:
            pinky_target[0] -= 1
        else:
            pinky_target[0] += 1

        #Adjusts for new target outside of boundaries
        pinky_target[0] = max(1, min(pinky_target[0], BOARD_SIZE[0]-2))
        pinky_target[1] = max(1, min(pinky_target[1], BOARD_SIZE[1]-3))

    pinky_target = tuple(pinky_target)
    pinky_position = tuple(ghost_positions['pinky'])
    path = astar(gameboard_state, pinky_position, pinky_target)
    if path and len(path) > 1:
        ghost_positions['pinky'] = list(path[1])

    #Inky
    inky_target = np.array(pacman_position['position'])
    blinky_position = np.array(list(ghost_positions['blinky']))
    if pacman_position['direction'] == 'up':
        inky_target = [inky_target[0]-2, inky_target[1]-2]
    elif pacman_position['direction'] == 'down':
        inky_target = [inky_target[0], inky_target[1]+2]
    elif pacman_position['direction'] == 'left':
        inky_target = [inky_target[0]-2, inky_target[1]]
    else:
        inky_target = [inky_target[0]+2, inky_target[1]]

    inky_target = (2*np.array(pacman_position['position']) - np.array(blinky_position))
    #Adjusts if target is out of boundaries
    inky_target[0] = max(1, min(inky_target[0], BOARD_SIZE[0]-2))
    inky_target[1] = max(1, min(inky_target[1], BOARD_SIZE[1]-3))

    #Adjusts if target is a wall
    while gameboard_state[(inky_target[0], inky_target[1])] in [0,3]:
        if pacman_position['direction']['up']:
            inky_target[1] -=1
        elif pacman_position['direction']['down']:
            inky_target[1] += 1
        elif pacman_position['direction']['left']:
            inky_target[0] -= 1
        else:
            inky_target[0] += 1

    inky_target = tuple(inky_target)
    inky_position = tuple(ghost_positions['inky'])
    path = astar(gameboard_state, inky_position, inky_target)
    if path and len(path) > 1:
        ghost_positions['inky'] = list(path[1])

    #Clyde
    clyde_position = tuple(ghost_positions['clyde'])
    path = astar(gameboard_state, clyde_position, pacman_position['position'])
    if path and len(path) > 9:
        ghost_positions['clyde'] = list(path[1])
    else:
        path = astar(gameboard_state, clyde_position, (1,28))
        if path and len(path) > 1:
            ghost_positions['clyde'] = list(path[1])

def check_collision():
    tuple_ghost_positions = [tuple(pos) for pos in ghost_positions.values()]
    if (pacman_position['position'] in tuple_ghost_positions):
        reset()

def main():
    running = True
    while running:
        draw_board()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    move_pacman('up')
                    move_ghosts()
                elif event.key == pygame.K_DOWN:
                    move_pacman('down')
                    move_ghosts()
                elif event.key == pygame.K_LEFT:
                    move_pacman('left')
                    move_ghosts()
                elif event.key == pygame.K_RIGHT:
                    move_pacman('right')
                    move_ghosts()
            if event.type == pygame.QUIT:
                running = False
        check_collision()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()