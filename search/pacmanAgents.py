# pacmanAgents.py
# ---------------
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


from pacman import Directions, GameState
from game import Agent
import random
import game
import util

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
