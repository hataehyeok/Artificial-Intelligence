def V(state, agent, depth):
      actions = state.getLegalActions(agent)

      if state.isWin() or state.isLose() or actions == [Directions.STOP]:
        return state.getScore(), Directions.STOP
      
      if depth == 0:
        return self.evaluationFunction(state), Directions.STOP
      
      Vs = [(V(state.generateSuccessor(agent, action),
               agent + 1 if agent < state.getNumAgents() - 1 else 0,
               depth if agent < state.getNumAgents() - 1 else depth - 1)[0], action)
            for action in actions if action != Directions.STOP]
      
      if agent == self.index:
        return max(Vs)
      else:
        return min(Vs)

    return V(gameState, self.index, self.depth)[1]


def V(state, agent, depth):
      actions = state.getLegalActions(agent)

      if state.isWin() or state.isLose() or actions == [Directions.STOP]:
        return state.getScore(), Directions.STOP
      
      if depth == 0:
        return self.evaluationFunction(state), Directions.STOP
      
      if agent < state.getNumAgents() - 1:
        newAgent = agent + 1
        newDepth = depth
      else:
        newAgent = 0
        newDepth = depth - 1

      if agent == self.index:
        return max((V(state.generateSuccessor(agent, action), newAgent, newDepth)[0], action)
                   for action in actions if action != Directions.STOP)
      else:
        return sum(V(state.generateSuccessor(agent, action), newAgent, newDepth)[0] / len(actions) for action in actions), None

    return V(gameState, self.index, self.depth)[1]



init = self.index
    d = self.depth
    agent_idx_max = gameState.getNumAgents() - 1

    # Definition of minimax of V
    def V_maxopp_odd(state, depth, agent_idx):
      legalMoves = state.getLegalActions(agent_idx)
      # Case of IsEnd(s)
      if (legalMoves == [Directions.STOP] or state.isWin() or state.isLose()):
        return (state.getScore(), Directions.STOP)
      # Case of d = 0
      if (depth == 0):
        return (self.evaluationFunction(state), Directions.STOP)
      # Consider minimax tree will have multiple min layers
      # h depth -> Make h(n+1) minimax tree
      if (agent_idx < agent_idx_max):
        next_agent = agent_idx + 1
        next_depth = depth
      else:
        next_agent = 0
        next_depth = depth - 1
      
      # Case of Player is pac-man or ghost
      if (agent_idx == init):
        valid_actions = [action for action in legalMoves if action != Directions.STOP]
        # For tracking action corresponding to value, use pair vector data structure
        value_action_pairs = [
            (V_maxopp_odd(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
            for action in valid_actions
        ]
        bestScore = max(value_action_pairs)
      else:
        # Odd case => min
        if (agent_idx % 2 == 1):
          valid_actions = [action for action in legalMoves if action != Directions.STOP]
          # For tracking action corresponding to value, use pair vector data structure
          value_action_pairs = [
              (V_maxopp_odd(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0], action)
              for action in valid_actions
          ]
          bestScore = min(value_action_pairs)
        # Even case => Expect
        else:
          temp = 0
          prob = 1.0 / len(legalMoves)
          for action in legalMoves:
            each_value = V_maxopp_odd(state.generateSuccessor(agent_idx, action), next_depth, next_agent)[0]
            expectied_value = each_value * prob
            temp += expectied_value
          bestScore = (temp, None)
        
        return bestScore

    return V_maxopp_odd(gameState.generateSuccessor(0, action), d, init + 1)[0]