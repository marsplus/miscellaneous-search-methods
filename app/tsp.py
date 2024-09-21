from __future__ import annotations

from typing import List

from app.states import State


class TSPState(State):
    def __init__(
        self,
        visited: List[int],
        current_city: int,
        dist: List[List[float]],
        num_city: int,
    ):
        """
        Initialize a Traveling Salesmen Problem (TSP).
        Args:
        - visited: list of cities visited in order.
        - current_city: the city we are currently at.
        - dist: the distance matrix.
        - N: total number of cities.
        """
        self.visited = visited
        self.current_city = current_city
        self.dist = dist
        self.N = num_city

    def is_terminal(self) -> bool:
        """Check if all cities have been visited"""
        return len(self.visited) == self.N

    def get_actions(self) -> List[int]:
        """
        Get available actions (next cities to visit) from the current state.
        """
        if self.is_terminal():
            return []
        return [city for city in range(self.N) if city not in self.visited]

    def next_state(self, action: int) -> TSPState:
        """
        Apply an action (visit a new city) and return the resulting new state.
        Args:
        - action: the city to visit next.

        Returns:
        - new_state: a new TSPState after applying the action.
        """
        new_visited = self.visited + [action]
        return TSPState(new_visited, action, self.dist, self.N)

    def total_cost(self) -> float:
        """Calculate the total cost (tour length) based on the visited cities."""
        total_cost = 0.0
        for i in range(1, len(self.visited)):
            total_cost += self.dist[self.visited[i - 1]][self.visited[i]]

        if self.is_terminal():
            total_cost += self.dist[self.visited[-1]][self.visited[0]]

        return total_cost

    def get_cost(self, state: State, action: int) -> float:
        if isinstance(state, TSPState):
            return state.dist[state.current_city][action]
        else:
            raise ValueError("The state passed to get_cost must be of type TSPState.")

    def get_current_state(self) -> State:
        return self
