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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        #food positions:
        foodsPoses = newFood.asList()
        # calculate all food distances:
        foodDists = [manhattanDistance(newPos, foodPos) for foodPos in foodsPoses]
        #ghost positions:
        closestGhostPos = newGhostStates[0].configuration.pos
        #ghost distances:
        closestGhostDist = manhattanDistance(newPos, closestGhostPos)
        #new and current score achieved:
        newScore = successorGameState.getScore()
        currScore = scoreEvaluationFunction(currentGameState)
        difference = newScore - currScore
        #number of new foods in succ state:
        newFoodNum = successorGameState.getNumFood()
        #least scared time:
        shortestScaredTime = min(newScaredTimes)

        if len(foodDists) == 0:
            closestFood = 0
        else:
            closestFood = min(foodDists)

        if shortestScaredTime != 0:
            closestGhostDist = -closestGhostDist * 3
        if action == 'Stop':
            evalValue = 1 / closestFood
        else:
            evalValue = (10 / (closestFood + 1)) + (closestGhostDist / 10) + difference + (10 / (newFoodNum + 1))

        return evalValue


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()

        def minimax(gameState, agentIndex, depth=0):
            bestAction = None
            # if we have reached a terminal state:
            if (depth == self.depth) or gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState)]
            elif agentIndex == (gameState.getNumAgents() - 1):  # if we have checked all agents
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1
            # MAX:
            if agentIndex == 0:  # because pacman agent is the max agent and first row in tree
                # initiliaze it to minus infinite:
                max = -float("inf")
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succState = gameState.generateSuccessor(agentIndex, action)
                    maxNew = minimax(succState, childAgentIndex, depth)[0]
                    if maxNew >= max:
                        max = maxNew
                        bestAction = action
                return max, bestAction
            # MIN:
            else:
                # initiliaze it to positive infinite:
                min = float("inf")
                legalActions = gameState.getLegalActions(agentIndex)
                for action in legalActions:
                    succState = gameState.generateSuccessor(agentIndex, action)
                    minNew = minimax(succState, childAgentIndex, depth)[0]
                    if minNew <= min:
                        min = minNew
                        bestAction = action
            return min, bestAction

        result = minimax(gameState, self.index)
        bestScore = result[0]
        bestMove = result[1]
        return bestMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        agentIndex = 0

        def max_value(state, depth, alpha, beta):
            maxScore = float("-inf")
            # check end:
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(agentIndex)
            score = maxScore
            bestAction = Directions.STOP
            for action in actions:
                score = min_value(state.generateSuccessor(agentIndex, action), depth, 1, alpha, beta)
                if score > maxScore:
                    maxScore = score
                    bestAction = action
                alpha = max(alpha, maxScore)
                if maxScore > beta:
                    return maxScore
            if depth == 0:
                return bestAction
            else:
                return maxScore

        def min_value(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            nextGhost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                nextGhost = agentIndex
            actions = state.getLegalActions(ghost)
            minScore = float("inf")
            score = minScore
            for action in actions:
                if nextGhost == agentIndex:
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_value(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_value(state.generateSuccessor(ghost, action), depth, nextGhost, alpha, beta)
                if score < minScore:
                    minScore = score
                beta = min(beta, minScore)
                if minScore < alpha:
                    return minScore
            return minScore

        return max_value(gameState, 0, float("-inf"), float("inf"))


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

        # util.raiseNotDefined()
        def expectimax(gameState, agentIndex, depth=0):
            bestAction = None
            # if we have reached a terminal state:
            if (depth == self.depth) or gameState.isWin() or gameState.isLose():
                return [self.evaluationFunction(gameState)]
            elif agentIndex == (gameState.getNumAgents() - 1):  # if we have checked all agents
                depth += 1
                childAgentIndex = self.index
            else:
                childAgentIndex = agentIndex + 1
            # if the player is max:
            if agentIndex == self.index:
                v = -float("inf")
            # if the player is chance:
            else:
                v = 0

            legalActions = gameState.getLegalActions(agentIndex)
            for action in legalActions:
                succState = gameState.generateSuccessor(agentIndex, action)
                expectedMax = expectimax(succState, childAgentIndex, depth)[0]
                if agentIndex == self.index:
                    if expectedMax > v:
                        v = expectedMax
                        bestAction = action
                else:
                    v = v + ((1/len(legalActions)) * expectedMax)
            return v, bestAction


        result = expectimax(gameState, self.index)
        bestScore = result[0]
        bestMove = result[1]
        return bestMove


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <Full explanation in report file.>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    #score achieved from food:

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    foodsPoses = food.asList()
    # calculate all food distances:
    foodDists = [manhattanDistance(pos, foodPos) for foodPos in foodsPoses]

    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()

    #calculate score achieved from FOOD:
    foodScore = 0
    #the less the food distance, the higher the evaluation function's result
    foodCalculations = [1.0/(manhattanDistance(pos, foodPos)) for foodPos in foodsPoses]
    if len(foodDists)==0:
        foodScore = 0
    else:
        foodScore = max(foodCalculations)

    #calculate score achieved from GHOSTS:
    ghostScore = 0
    #the further the ghosts, the higher the evaluation function's result
    for ghost in ghostStates:
        dist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer>0:
            if dist < 5:
                ghostScore += dist * 10
            else:
                ghostScore += dist
        else:
            if dist < 5:
                ghostScore -= dist * 10
            else:
                ghostScore -= dist

    #calculate score achieved from CAPSULES:
    capsuleScore = 0
    #the closer the capsules, the higher the evaluation function's result
    capsuleDistsInverse = [10.0/manhattanDistance(pos, capsule) for capsule in capsuleList]
    if len(capsuleDistsInverse)==0:
        capsuleScore = 0
        capsuleScore = 0
    else:
        capsuleScore = max(capsuleDistsInverse)

    totalScore = currentGameState.getScore() + foodScore + ghostScore + capsuleScore
    return totalScore

# Abbreviation
better = betterEvaluationFunction
