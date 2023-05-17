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
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxmin(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
            for action in valid_actions
        ]
        bestScore = min(value_action_pairs)
      
      return bestScore

    return V_maxmin(gameState, d, init)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the minimax Q-Value from the current gameState and given action
      using self.depth and self.evaluationFunction.
      Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves.
    """
    # BEGIN_YOUR_ANSWER
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    #Definition of minimax of V
    def V_maxmin(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
            for action in valid_actions
        ]
        bestScore = min(value_action_pairs)
      
      return bestScore
    return V_maxmin(gameState.generateSuccessor(0, action), d, init)[0]

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
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxopp(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxopp(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        temp = 0
        prob = 1.0 / len(legalMoves)
        for action in legalMoves:
          each_value = V_maxopp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
          expectied_value = each_value * prob
          temp += expectied_value
        bestScore = (temp, None)
      
      return bestScore

    return V_maxopp(gameState, d, init)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxopp(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxopp(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        temp = 0
        prob = 1.0 / len(legalMoves)
        for action in legalMoves:
          each_value = V_maxopp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
          expectied_value = each_value * prob
          temp += expectied_value
        bestScore = (temp, None)
      
      return bestScore

    return V_maxopp(gameState.generateSuccessor(0, action), d, init + 1)[0]
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
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxopp_bias(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxopp_bias(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1
        temp = 0
        _biasStop = 0.5 + (0.5 / len(legalMoves))
        _bias = 0.5 / len(legalMoves)
        for action in legalMoves:
          prob = _biasStop if action == Directions.STOP else _bias
          each_value = V_maxopp_bias(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
          expectied_value = each_value * prob
          temp += expectied_value
        bestScore = (temp, None)
      
      return bestScore

    return V_maxopp_bias(gameState, d, init)[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the biased-expectimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxopp_bias(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxopp_bias(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1
        temp = 0
        _biasStop = 0.5 + (0.5 / len(legalMoves))
        _bias = 0.5 / len(legalMoves)
        for action in legalMoves:
          prob = _biasStop if action == Directions.STOP else _bias
          each_value = V_maxopp_bias(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
          expectied_value = each_value * prob
          temp += expectied_value
        bestScore = (temp, None)
      
      return bestScore

    return V_maxopp_bias(gameState.generateSuccessor(0, action), d, init + 1)[0]
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
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxmin_exp(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin_exp(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        if (agent_idx % 2 == 1):
          valid_actions = [action for action in legalMoves]
          # For tracking action corresponding to value, use pair vector data structure
          value_action_pairs = [
              (V_maxmin_exp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
              for action in valid_actions
          ]
          bestScore = min(value_action_pairs)
        else:
          temp = 0
          prob = 1.0 / len(legalMoves)
          for action in legalMoves:
            each_value = V_maxmin_exp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
            expectied_value = each_value * prob
            temp += expectied_value
          bestScore = (temp, None)
      
      return bestScore

    return V_maxmin_exp(gameState, d, init)[1]
    
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxmin_exp(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxmin_exp(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        if (agent_idx % 2 == 1):
          valid_actions = [action for action in legalMoves]
          # For tracking action corresponding to value, use pair vector data structure
          value_action_pairs = [
              (V_maxmin_exp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
              for action in valid_actions
          ]
          bestScore = min(value_action_pairs)
        else:
          temp = 0
          prob = 1.0 / len(legalMoves)
          for action in legalMoves:
            each_value = V_maxmin_exp(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
            expectied_value = each_value * prob
            temp += expectied_value
          bestScore = (temp, None)
      
      return bestScore

    return V_maxmin_exp(gameState.generateSuccessor(0, action), d, init + 1)[0]
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
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxmin_exp_prun(state, depth, agent_idx, a, b):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers

      if (agent_idx == init):
        valid_actions = [action for action in legalMoves if action != Directions.STOP]
        # For tracking action corresponding to value, use pair vector data structure
        _vaild = (-float('inf'), None)
        for action in valid_actions:
          score = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1, a, b)[0]
          _vaild = max(_vaild, (score, action))
          a = max(_vaild[0], a)
          if a > b:
            break
        return _vaild
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        if (agent_idx % 2 == 1):
          valid_actions = [action for action in legalMoves if action != Directions.STOP]
          _vaild = (float('inf'),None)
          for action in valid_actions:
            score = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1, a, b)[0]
            _vaild = min(_vaild, (score, action))
            a = min(_vaild[0], a)
            if a > b:
              break
          return _vaild
        else:
          temp = 0
          prob = 1.0 / len(legalMoves)
          for action in legalMoves:
            each_value = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), next_depth, next_agent, a, b)[0]
            expectied_value = each_value * prob
            temp += expectied_value
          bestScore = (temp, None)
      
      return bestScore

    return V_maxmin_exp_prun(gameState, d, init, -float('inf'), float('inf'))[1]
    # END_YOUR_ANSWER
  
  def getQ(self, gameState, action):
    """
      Returns the expectiminimax Q-Value using self.depth and self.evaluationFunction.
    """
    # BEGIN_YOUR_ANSWER
    init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxmin_exp_prun(state, depth, agent_idx, a, b):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers

      if (agent_idx == init):
        valid_actions = [action for action in legalMoves if action != Directions.STOP]
        # For tracking action corresponding to value, use pair vector data structure
        _vaild = (-float('inf'), None)
        for action in valid_actions:
          score = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1, a, b)[0]
          _vaild = max(_vaild, (score, action))
          a = max(_vaild[0], a)
          if a > b:
            break
        return _vaild
      else:
        if (agent_idx < agent_idx_max):
          next_agent = agent_idx + 1
          next_depth = depth
        else:
          next_agent = 0
          next_depth = depth - 1

        if (agent_idx % 2 == 1):
          valid_actions = [action for action in legalMoves if action != Directions.STOP]
          _vaild = (float('inf'),None)
          for action in valid_actions:
            score = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), depth, agent_idx + 1, a, b)[0]
            _vaild = min(_vaild, (score, action))
            a = min(_vaild[0], a)
            if a > b:
              break
          return _vaild
        else:
          temp = 0
          prob = 1.0 / len(legalMoves)
          for action in legalMoves:
            each_value = V_maxmin_exp_prun(state.generateSuccessor(agent_idx, action), next_depth, next_agent, a, b)[0]
            expectied_value = each_value * prob
            temp += expectied_value
          bestScore = (temp, None)
      
      return bestScore

    return V_maxmin_exp_prun(gameState.generateSuccessor(0, action), d, init + 1, -float('inf'), float('inf'))[0]
    # END_YOUR_ANSWER

######################################################################################
# Problem 6a: creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
  Your extreme, unstoppable evaluation function (problem 6).
  """

  # BEGIN_YOUR_ANSWER
  # Eval(s, a) = score + dist from G + 10*T/dist from S + 5/dist from F + -1*number of food + 10/dist from C
  PacmanPosition = currentGameState.getPacmanPosition()
  score = currentGameState.getScore()
  numFood = currentGameState.getNumFood()

  disfromGhosts = []
  distfromScare = []
  for ghost in currentGameState.getGhostStates():
    if (ghost.scaredTimer > 0):
      distfromScare.append(ghost.scaredTimer / manhattanDistance(PacmanPosition, ghost.getPosition()))
    else:
      disfromGhosts.append(manhattanDistance(PacmanPosition, ghost.getPosition()))
  
  if (len(disfromGhosts) > 0):
    score += min(disfromGhosts)
  if (len(distfromScare) > 0):
    score += 10 * min(distfromScare)

  score += (-1) * numFood

  distfromFoods = [manhattanDistance(PacmanPosition, foodPosition) for foodPosition in currentGameState.getFood().asList()]
  if (len(distfromFoods) > 0):
    score += 5 * 1.0/min(distfromFoods)

  distfromC = [1 / manhattanDistance(PacmanPosition, capsulePosition) for capsulePosition in currentGameState.getCapsules()]
  if (len(distfromC) > 0):
    score += 10 * min(distfromC)

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
  return 'ExpectiminimaxAgent'  # remove this line before writing code
  # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction
