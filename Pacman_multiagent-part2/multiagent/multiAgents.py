# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        food_list = newFood.asList()
        food_distance = []
        value = successorGameState.getScore()

        #calculating distance between pacman and the ghosts
        near_ghost = 0 #distance to nearest ghostState                                  #newGhostStates: list, newGhostStates[0]: instance 
        distance_to_ghost = manhattanDistance(newPos, newGhostStates[0].getPosition())  #newGhostStates[0].getPosition(): tuple, which represents the state(position) of the new ghost
        if distance_to_ghost != 0:
          value -= 1/distance_to_ghost  #the reciprocal number (divide 1 by the distance)
          if distance_to_ghost <= 1:  #if ghost gets too close, move further
            near_ghost += 1
        
        #calculating distance to the closest food
        for food in food_list:
          food_distance.append(manhattanDistance(newPos, food))
        if len(food_distance):  #if food_distance is not empty: there's still food!
          value += 1/float(min(food_distance))

        return value - near_ghost


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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        value = self.Minimax_decision(gameState, 0, 0)
        return value[0] #return the action


    def Minimax_decision(self, gameState, currentAgentIndex, currentDepth):
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0 #pacman = MAX
        currentDepth += 1 #further down the tree

      if currentDepth == self.depth:
        return self.evaluationFunction(gameState)

      if currentAgentIndex == 0:  # currentAgentIndex=0 means pacman's index
       return self.max_value(gameState, currentAgentIndex, currentDepth)  #Max is pacman
      else: #while currentAgentIndex >= 1 means ghosts' indices
        return self.min_value(gameState, currentAgentIndex, currentDepth) #Min are ghosts


    def max_value(self, gameState, currentAgentIndex, currentDepth):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   
      
      n = (" ", float("-inf"))  # n = -oo

      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        maxvalue = self.Minimax_decision(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth) 
        if type(maxvalue) is tuple:
          maxvalue = maxvalue[1]

        if maxvalue > n[1]:
          n = (action, max(n[1], maxvalue))

      return n

    def min_value(self, gameState, currentAgentIndex, currentDepth):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   

      n = (" ", float("inf"))  # n = +oo

      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        minvalue = self.Minimax_decision(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth)
        if type(minvalue) is tuple:
          minvalue = minvalue[1]

        if minvalue < n[1]:
          n = (action, min(n[1], minvalue))

      return n

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = float("-inf")
        beta = float("inf")
        value = self.AlphaBetaPruning(gameState, 0, 0, alpha, beta)
        return value[0] #return the action

    def AlphaBetaPruning(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0 #pacman = MAX
        currentDepth += 1 #further down the tree

      if currentDepth == self.depth:
        return self.evaluationFunction(gameState)

      if currentAgentIndex == 0:  # currentAgentIndex=0 means pacman's index
       return self.max_value_AB_pruning(gameState, currentAgentIndex, currentDepth, alpha, beta)  #Max is pacman
      else: #while currentAgentIndex >= 1 means ghosts' indices
        return self.min_value_AB_pruning(gameState, currentAgentIndex, currentDepth, alpha, beta) #Min are ghosts

    def max_value_AB_pruning(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   

      n = (" ", float("-inf"))  # n = -oo
      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        maxvalue = self.AlphaBetaPruning(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth, alpha, beta)
        if type(maxvalue) is tuple:
          maxvalue = maxvalue[1]

        if maxvalue > n[1]:
          n = (action, max(n[1], maxvalue))

        if n[1] > beta:
          return n

        alpha = max(alpha, n[1])

      return n

    def min_value_AB_pruning(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   

      n = (" ", float("inf"))  # n = +oo
      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        minvalue = self.AlphaBetaPruning(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth, alpha, beta)
        if type(minvalue) is tuple:
          minvalue = minvalue[1]

        if minvalue < n[1]:
          n = (action, min(n[1], minvalue))

        if n[1] < alpha:
          return n

        beta = min(beta, n[1])

      return n




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        value = self.Expectimax(gameState, 0, 0)
        return value[0] #return the action

    def Expectimax(self, gameState, currentAgentIndex, currentDepth):
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0 #pacman = MAX
        currentDepth += 1 #further down the tree

      if currentDepth == self.depth:
        return self.evaluationFunction(gameState)

      if currentAgentIndex == 0:  # currentAgentIndex=0 means pacman's index
       return self.max_value_expectimax(gameState, currentAgentIndex, currentDepth)  #Max is pacman
      else: #while currentAgentIndex >= 1 means ghosts' indices
        return self.expected_value(gameState, currentAgentIndex, currentDepth) #Min are ghosts

    def max_value_expectimax(self, gameState, currentAgentIndex, currentDepth):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   

      n = (" ", float("-inf"))  # n = -oo
      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        maxvalue = self.Expectimax(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth)
        if type(maxvalue) is tuple:
          maxvalue = maxvalue[1]

        if maxvalue > n[1]:
          n = (action, max(n[1], maxvalue))

      return n

    def expected_value(self, gameState, currentAgentIndex, currentDepth):
      if not gameState.getLegalActions(currentAgentIndex) or gameState.isWin() or gameState.isLose():  #if reached terminate state, utility(s)
        return self.evaluationFunction(gameState)   

      n_value = 0.0
      n_action = " "
      probability = 1.0/len(gameState.getLegalActions(currentAgentIndex))
      for action in gameState.getLegalActions(currentAgentIndex):
        if action == Directions.STOP:
          continue;

        expectvalue = self.Expectimax(gameState.generateSuccessor(currentAgentIndex, action), currentAgentIndex+1, currentDepth)
        if type(expectvalue) is tuple:
          expectvalue = expectvalue[1]

        n_value += expectvalue * probability
        n_action = action

      n = (n_action, n_value)

      return n

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghostState.scaredTimer for ghostState in ghost_states]

    food_list = food.asList()
    ghost_distance = []
    food_distance = []

    value = currentGameState.getScore()

    #calculating distance between pacman and the ghosts
    near_ghost = 0  #distance to nearest ghostState 
    for ghost in ghost_states:                                                      #newGhostStates: list, newGhostStates[0]: instance 
        ghost_distance = manhattanDistance(position, ghost_states[0].getPosition()) #newGhostStates[0].getPosition(): tuple, which represents the state(position) of the new ghost
        if ghost_distance > 1:
            if ghost.scaredTimer > 0:   #ghost is scared, go to eat him
                value += 5 / float(ghost_distance)  #the value to eat a scared ghost is little greater than the value of food
            else: #ghost aren't scared, move away from them
                value -= 1 / float(ghost_distance)
        elif ghost_distance <= 1:   #if ghost gets too close, move further
          near_ghost += 1

    #calculating distance to the closest food
    for food in food_list:
      food_distance.append(manhattanDistance(position, food))
    if len(food_distance):  #if food_distance is not empty: there's still food!
        value += 1 / float(min(food_distance))

    return value - near_ghost


# Abbreviation
better = betterEvaluationFunction