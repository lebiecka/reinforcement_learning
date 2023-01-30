class World:
    def __init__(self, size, picker_position,  mushroom_position):
        self.size = size
        self.mushroom_position = mushroom_position
        self.picker_position = picker_position
        self.mushroom_in_car = False

    def get_number_of_states(self):
        return self.size*self.size*self.size*self.size
    
    def get_state(self):
        """
        Returns:
            State of the environment
        """
        state = self.picker_position[0]*self.size*self.size*self.size
        state = state + self.picker_position[1]*self.size*self.size
        state = state + self.mushroom_position[0]*self.size
        state = state + self.mushroom_position[1]

        return state

    def step(self, action):
        """ Moves into the next time step
        Args:
            action: Action performed by the agent
        Returns:
            Reward for this action
        """
        
        (x, y) = self.picker_position
        
        if action == 0:  # Go South
            if y == self.size - 1:
                return -10, False
            else:
                self.picker_position = (x, y + 1)
                return -1, False
        elif action == 1:  # Go North
            if y == 0:
                return -10, False
            else:
                self.picker_position = (x, y - 1)
                return -1, False
        elif action == 2:  # Go East
            if x == 0:
                return -10, False
            else:
                self.picker_position = (x - 1, y)
                return -1, False
        elif action == 3:  # Go West
            if x == self.size - 1:
                return -10, False
            else:
                self.picker_position = (x + 1, y)
                return -1, False
        elif action == 4:  # Pickup item
            if self.mushroom_position != (x, y):
                return -10, False
            else:
                self.mushroom_in_car = True
                return 20, True