from util import manhattanDistance
from game import Directions
import random, util

from game import Agent



class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None

  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """
  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)
    
      gameState.isWin():
        Returns True if it's a winning state
    
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """
    # BEGIN_YOUR_ANSWER
    def minimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = minimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: min
      else:
        value = float('inf')
        best_action = None
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        
        
        for action in gameState.getLegalActions(agent):
          next_value, _ = minimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          if next_value < value:
            value = next_value
            best_action = action
        return value, best_action

    return minimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    def minimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = minimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: min
      else:
        value = float('inf')
        best_action = None
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        for action in gameState.getLegalActions(agent):
          next_value, _ = minimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          if next_value < value:
            value = next_value
            best_action = action
        return value, best_action
      
    return minimax(1, self.depth, gameState.generateSuccessor(0, action))[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = expectimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: expect
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        value = 0
        legal_actions = gameState.getLegalActions(agent)
        prob = 1.0 / len(legal_actions)
        for action in legal_actions:
          next_value, _ = expectimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          value += prob * next_value
        return value, None

    return expectimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = expectimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: expect
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        value = 0
        legal_actions = gameState.getLegalActions(agent)
        prob = 1.0 / len(legal_actions)
        for action in legal_actions:
          next_value, _ = expectimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          value += prob * next_value
        return value, None
        
    return expectimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing biased-expectimax

class BiasedExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your biased-expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the biased-expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing stop-biasedly from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def biased_expectimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = biased_expectimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: biased expect
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        value = 0
        legal_actions = gameState.getLegalActions(agent)
        stop_bias = 0.5 + 0.5 / len(legal_actions)
        for action in legal_actions:
          prob = stop_bias if action == Directions.STOP else 0.5 / len(legal_actions)
          next_value, _ = biased_expectimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          value += prob * next_value
        return value, None

    return biased_expectimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def biased_expectimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = biased_expectimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: biased expect
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1
        value = 0
        legal_actions = gameState.getLegalActions(agent)
        stop_bias = 0.5 + 0.5 / len(legal_actions)
        for action in legal_actions:
          prob = stop_bias if action == Directions.STOP else 0.5 / len(legal_actions)
          next_value, _ = biased_expectimax(next_agent, depth, gameState.generateSuccessor(agent, action))
          value += prob * next_value
        return value, None

    return biased_expectimax(1, self.depth, gameState.generateSuccessor(0, action))
    # END_YOUR_ANSWER

######################################################################################
# Problem 4a: implementing expectiminimax

class ExpectiminimaxAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent (problem 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction

      The even-numbered ghost should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_ANSWER
    def expectiminimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = expectiminimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: expect or min
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1

        legal_actions = gameState.getLegalActions(agent)

        #odd: min
        if agent % 2 == 1:
          value = float('inf')
          for action in legal_actions:
            next_value, _ = expectiminimax(next_agent, depth, gameState.generateSuccessor(agent, action))
            value = min(value, next_value)
          return value, None

        #even: expect
        else:
          value = 0
          prob = 1.0 / len(legal_actions)
          for action in legal_actions:
            next_value, _ = expectiminimax(next_agent, depth, gameState.generateSuccessor(agent, action))
            value += prob * next_value
          return value, None

    return expectiminimax(0, self.depth, gameState)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def expectiminimax(agent, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = expectiminimax(1, depth, gameState.generateSuccessor(agent, action))
          if next_value > value:
            value = next_value
            best_action = action
        return value, best_action

      #Ghost: expect or min
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1

        legal_actions = gameState.getLegalActions(agent)

        #odd: min
        if agent % 2 == 1:
          value = float('inf')
          for action in legal_actions:
            next_value, _ = expectiminimax(next_agent, depth, gameState.generateSuccessor(agent, action))
            value = min(value, next_value)
          return value, None

        #even: expect
        else:
          value = 0
          prob = 1.0 / len(legal_actions)
          for action in legal_actions:
            next_value, _ = expectiminimax(next_agent, depth, gameState.generateSuccessor(agent, action))
            value += prob * next_value
          return value, None

    return expectiminimax(1, self.depth, gameState.generateSuccessor(0, action))[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 5a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your expectiminimax agent with alpha-beta pruning (problem 5)
  """

  def getAction(self, gameState):
    """
      Returns the expectiminimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_ANSWER
    def alpha_beta(agent, depth, gameState, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = alpha_beta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta)
          if next_value > value:
            value = next_value
            best_action = action
          alpha = max(alpha, value)
          if alpha >= beta:
            break
        return value, best_action

      #Ghost: expect or min
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1

        legal_actions = gameState.getLegalActions(agent)

        #odd: min
        if agent % 2 == 1:
          value = float('inf')
          best_action = None
          for action in legal_actions:
            next_value, _ = alpha_beta(next_agent, depth, gameState.generateSuccessor(agent, action), alpha, beta)
            if next_value < value:
              value = next_value
              best_action = action
            beta = min(beta, value)
            if alpha >= beta:
              break
          return value, best_action

        #even: expect
        else:
          value = 0
          prob = 1.0 / len(legal_actions)
          for action in legal_actions:
            next_value, _ = alpha_beta(next_agent, depth, gameState.generateSuccessor(agent, action), alpha, beta)
            value += prob * next_value
          return value, None

    return alpha_beta(0, self.depth, gameState, -float('inf'), float('inf'))[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    def alpha_beta(agent, depth, gameState, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState), None

      #Pacman: max
      if agent == 0:
        value = -float('inf')
        best_action = None
        for action in gameState.getLegalActions(agent):
          next_value, _ = alpha_beta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta)
          if next_value > value:
            value = next_value
            best_action = action
          alpha = max(alpha, value)
          if alpha >= beta:
            break
        return value, best_action

      #Ghost: expect or min
      else:
        next_agent = agent + 1
        if next_agent == gameState.getNumAgents():
          next_agent = 0
          depth -= 1

        legal_actions = gameState.getLegalActions(agent)

        #odd: min
        if agent % 2 == 1:
          value = float('inf')
          best_action = None
          for action in legal_actions:
            next_value, _ = alpha_beta(next_agent, depth, gameState.generateSuccessor(agent, action), alpha, beta)
            if next_value < value:
              value = next_value
              best_action = action
            beta = min(beta, value)
            if alpha >= beta:
              break
          return value, best_action

        #even: expect
        else:
          value = 0
          prob = 1.0 / len(legal_actions)
          for action in legal_actions:
            next_value, _ = alpha_beta(next_agent, depth, gameState.generateSuccessor(agent, action), alpha, beta)
            value += prob * next_value
          return value, None
        
    return alpha_beta(1, self.depth, gameState.generateSuccessor(0, action), -float('inf'), float('inf'))[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  # Retrieve relevant information
  pacman_pos = currentGameState.getPacmanPosition()
  ghost_states = currentGameState.getGhostStates()
  food = currentGameState.getFood()
  capsules = currentGameState.getCapsules()

  # Calculate distances to all ghosts and food
  ghost_distances = [manhattanDistance(pacman_pos, ghost.getPosition()) for ghost in ghost_states]
  food_distances = [manhattanDistance(pacman_pos, food_pos) for food_pos in food.asList()]
  capsule_distances = [manhattanDistance(pacman_pos, capsule) for capsule in capsules]

  # Define score
  score = currentGameState.getScore()

  # Feature 1: Closest food
  if food_distances:
    closest_food = min(food_distances)
    score += 2.0 / (closest_food + 1)

  # Feature 2: Number of capsules
  score -= 10.0 * len(capsules)

  # Feature 3: Remaining food
  remaining_food = len(food_distances)
  score -= 4.0 * remaining_food

  # Feature 4: Ghost distances and scared time
  num_scared_ghosts = 0
  for ghost_dist, ghost_state in zip(ghost_distances, ghost_states):
    scared_time = ghost_state.scaredTimer
    if ghost_dist <= 1:
      if scared_time > 0:
        score += 250
        num_scared_ghosts += 1
      else:
        score -= 250
    elif scared_time > 0:
      score += 15.0 / (ghost_dist + 1) * (scared_time / 5.0)

  # Feature 5: Distance to the nearest capsule
  if capsule_distances:
    nearest_capsule_distance = min(capsule_distances)
    score += 5.0 / (nearest_capsule_distance + 1)

  # Feature 6: Number of scared ghosts
  score += 50 * num_scared_ghosts

  return score
  # END_YOUR_ANSWER

def choiceAgent():
  """
    Choose the pacman agent model you want for problem 6.
    You can choose among the agents above or design your own agent model.
    You should return the name of class of pacman agent.
    (e.g. 'MinimaxAgent', 'BiasedExpectimaxAgent', 'MyOwnAgent', ...)
  """
  # BEGIN_YOUR_ANSWER
  return 'ExpectimaxAgent'
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction