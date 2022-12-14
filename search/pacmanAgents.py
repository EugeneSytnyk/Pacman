from pacman import Directions, GameState
from game import Agent

MAX_DEPTH = 7

def scoreEvaluation(state):
    return state.getScore()

class MinimaxAlphaBetaAgent(Agent):
    def minimax(self, state, agent, alpha, beta, depth):
        if state.isLose() or state.isWin() or depth == MAX_DEPTH:
            return [scoreEvaluation(state)]
        if agent == 0:
            bestValue = -float("inf")
            actions = state.getLegalActions(agent)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            for action in actions:
                nextState = state.generateSuccessor(agent, action)
                value = self.minimax(nextState, agent + 1, alpha, beta, depth + 1)[0]
                if value > bestValue:
                    bestValue = value
                    bestAction = action
                alpha = max(alpha, bestValue)
                if beta <= alpha:
                    break
            return [bestValue, bestAction]
        else:
            bestValue = float("inf")
            next_agent = agent + 1
            if agent == state.getNumAgents() - 1:
                next_agent = 0
            actions = state.getLegalActions(agent)

            for action in actions:
                nextState = state.generateSuccessor(agent, action)
                value = self.minimax(nextState, next_agent, alpha, beta, depth + 1)[0]
                if value < bestValue:
                    bestValue = value
                    bestAction = action
                beta = min(beta, bestValue)
                if beta <= alpha:
                    break
            return [bestValue, bestAction]

    def getAction(self, gameState: GameState):
        return self.minimax(gameState, self.index, -float("inf"), float("inf"), 0)[1]

class ExpectimaxAgent(Agent):
    def expectimax(self, state, agent, depth):
        if state.isLose() or state.isWin() or depth == MAX_DEPTH:
            return [scoreEvaluation(state)]
        if agent == 0:
            bestValue = -float("inf")
            actions = state.getLegalActions(agent)
            if Directions.STOP in actions:
                actions.remove(Directions.STOP)
            for action in actions:
                nextState = state.generateSuccessor(agent, action)
                value = self.expectimax(nextState, agent + 1, depth + 1)[0]
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            return [bestValue, bestAction]
        else:
            bestValue = 0
            next_agent = agent + 1
            if agent == state.getNumAgents() - 1:
                next_agent = 0
            actions = state.getLegalActions(agent)

            for action in actions:
                nextState = state.generateSuccessor(agent, action)
                bestValue += self.expectimax(nextState, next_agent, depth + 1)[0]
            bestValue = bestValue / len(actions)
            return [bestValue]

    def getAction(self, gameState: GameState):
        return self.expectimax(gameState, self.index, 0)[1]
