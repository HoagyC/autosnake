import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)

agents = []
grid_size = 35
timestep = 0

def create_grid(size):
    probs = np.array([0.9, 0.1/3, 0.1/3, 0.1/3])
    values = np.array([0,1,2,3], dtype=np.int8)
    grid = np.random.choice(values, (size,size), p=probs)
    return grid

def create_agents(number):
    for i in range(number):
        start_location = (np.random.randint(grid_size), np.random.randint(grid_size))
        Agent(start_snake = [], start_location=start_location)

def main():
    grid = create_grid(grid_size)
    create_agents(10)
    for step_number in range(10):
        print(grid)
        step(grid, agents)
    
def add_agent(agent, name=None):
    agents.append(agent)
    return len(agents) - 1

def delete_agent(grid, agent):
    for position in agent.positions[-(len(agent.snake) + 1):]:
        grid[position] = 0
    agents.remove(agent)

def integer_floor(x, base):
    return int(base * math.floor(x / base))

def get_random_direction():
    possible_directions = [(0,1),(1,0),(-1,0),(0,-1)]
    direction = possible_directions[np.random.randint(0,4)]
    return direction

def step(grid, agents):
    oldgrid = np.copy(grid)
    for agent in agents:
        prev_position = agent.positions[-1]
        current_position = agent.get_next_pos(grid)
        for position, value in zip(agent.positions[::-1], [-1] + agent.snake + [-5]):
            grid[position] = value + 5
        position_block = oldgrid[current_position]
        if position_block in {4,5,6,7}:
            delete_agent(grid, agent)
            grid[current_position] = 4
        if position_block in {1,2,3}:
            agent.add_block(position_block - 1)
            grid[current_position] = 0


class Agent():
    def __init__(self, start_location, start_snake = [], name=None):
        # nonlocal timestep
        self.id = add_agent(self, name)
        self.snake = start_snake
        self.initstep = timestep
        self.positions = [start_location]
        self.gene_length = 5
        self.direction = get_random_direction()
        self.transition_tensor = self.get_new_transition(start_snake)

    def add_block(self, block):
        self.snake.append(block)
        if len(self.snake) % 5 == 0:
            self.update_transition()
    
    def get_genes(self, snake):
        num_genes = integer_floor(len(self.snake), self.gene_length)
        for i in range(0, num_genes, self.gene_length):
            yield self.snake[i:i + self.gene_length]

    def get_next_pos(self, grid):
        direction = self.direction
        assert (direction in {(1,0), (-1,0), (0,-1), (0,1)})
        theta = -math.atan2(direction[0], direction[1]) + (math.pi / 2)
        direction_matrix = np.array([[int(math.cos(theta)),int(math.sin(theta))], [int(-math.sin(theta)), int(math.cos(theta))]])
        visual_field = np.array([[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1],[-1,2],[0,2],[1,2]])
        rotated_visual_field =  visual_field @ direction_matrix
        relative_visual_field = rotated_visual_field + self.positions[-1]
        local_area = []
        for i in relative_visual_field:
            local_area.append(grid[(i[0] % grid_size, i[1] % grid_size)])
        local_area = np.reshape(local_area, (3,3))
        decision = np.zeros(3)
        coord_shape = [(x, y) for x in range(3) for y in range(3)]
        for x, y in coord_shape:
            decision += self.transition_tensor[x,y,local_area[(x,y)],:]
        decision_result = np.argmax(decision)
        turn_key =[[1,0],[0,1],[0,-1]]
        base_direction = turn_key[decision_result]
        true_direction = base_direction @ direction_matrix
        # Movement testing
        # print(theta)
        # print('position:', self.positions[-1])
        # print('direction:', direction)
        # print('decision:', decision)
        # print('base_direction:', base_direction)
        # print('true_direction:', true_direction)
        new_pos = tuple((true_direction + self.positions[-1]) % grid_size)
        self.direction = tuple(true_direction)
        self.positions.append(new_pos)
        return new_pos

    def update_transition(self):
        assert(len(self.snake) % 5 == 0)
        gene = self.snake[-5:]
        pos_x = gene[0]
        pos_y = gene[1]
        obj = gene[2] * gene[3]
        direc = gene[4]
        self.transition_tensor[pos_x, pos_y, obj, direc] += 1

    def get_new_transition(self, snake):
        transition_tensor = np.zeros((3,3,9,3))
        genes = self.get_genes(snake)
        for gene in genes:
            pos_x = gene[0]
            pos_y = gene[1]
            obj = gene[2] * gene[3]
            direc = gene[4]
            transition_tensor[pos_x, pos_y, obj, direc] += 1
        return transition_tensor
        

if __name__ == "__main__":
    main()

