from abc import ABC, abstractmethod
from random import choice

from app.states import State


class Search(ABC):
    """
    A base meta-class for different search strategies.
    """

    def __init__(self, state: State):
        """
        Initialize the search with a given state.
        Args:
        - state: The initial state of the problem (any state that inherits from State).
        """
        self.state = state

    def update_state(self, state: State) -> None:
        self.state = state

    @abstractmethod
    def search(self) -> int:
        """
        Abstract method to perform the search.

        Returns:
        - next_state: The next state based on the specific search strategy.
        """
        pass

    def rollout(self, state: State) -> float:
        """
        Perform a random simulation from the current state to a terminal state.
        Args:
        - state: The current state to start the rollout from.

        Returns:
        - total_cost: The total cost from the start state to the terminal state.
        """
        current_state = state
        while not current_state.is_terminal():
            actions = current_state.get_actions()
            if actions:
                next_action = choice(actions)
                current_state = current_state.next_state(next_action)

        return current_state.total_cost()


class OneStepLookahead(Search):
    def search(self) -> int:
        best_action = None
        min_cost = float("inf")
        current_state = self.state.get_current_state()
        for action in self.state.get_actions():
            immediate_cost = self.state.get_cost(current_state, action)
            next_state = self.state.next_state(action)
            future_cost_estimate = self.rollout(next_state)
            total_cost = immediate_cost + future_cost_estimate

            if total_cost < min_cost:
                min_cost = total_cost
                best_action = action
        assert best_action is not None
        return best_action
