from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class State(ABC):
    """
    A base class representing the state of a dynamic programming problem.
    """

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the state is terminal (end of the problem)."""
        pass

    @abstractmethod
    def get_actions(self) -> List[int]:
        """Get the available actions (next possible states)."""
        pass

    @abstractmethod
    def next_state(self, action: int) -> State:
        """Return the next state after applying the action."""
        pass

    @abstractmethod
    def total_cost(self) -> float:
        """Return the total cost from the start state to the terminal state."""
        pass

    @abstractmethod
    def get_cost(self, state: State, action: int) -> float:
        """Return the cost of selecting `action` at `state`"""
        pass

    @abstractmethod
    def get_current_state(self) -> State:
        "Return the current state"
        pass
