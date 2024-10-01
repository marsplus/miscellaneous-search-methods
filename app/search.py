from __future__ import annotations

import math
import multiprocessing
import random
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple

from app.states import State


def simulate_rollout(state: State, max_depth: int) -> float:
    """
    Perform a random rollout simulation starting from the given state.

    Args:
    - state: The initial state for the simulation.
    - max_depth: the maximum depth to search

    Returns:
    - total_cost: The total cost accumulated during the simulation.
    """
    random.seed()
    current_state = state
    depth = 0
    while not current_state.is_terminal() and depth < max_depth:
        actions = current_state.get_actions()
        if actions:
            action = random.choice(actions)
            current_state = current_state.next_state(action)
        else:
            break
        depth += 1
    return current_state.total_cost()


class Search(ABC):
    """
    A base meta-class for different search strategies.
    """

    def __init__(
        self,
        state: State,
        num_sim: int = 100,
        num_processes: Optional[int] = None,
        max_depth: int = 20,
    ):
        """
        Initialize the search with a given state.
        Args:
        - state: The initial state of the problem (any state that inherits from State).
        - num_sim: the number of simulations to run
        """
        self.state = state
        self.num_sim = num_sim
        self.num_processes = num_processes if num_processes is not None else 4
        cpu_count = multiprocessing.cpu_count()
        max_processes = max(1, cpu_count - 1)
        self.num_processes = min(self.num_processes, max_processes)
        self.max_depth = max_depth

    def update_state(self, state: State) -> None:
        """Update the current state with a new state."""
        self.state = state

    @abstractmethod
    def search(self) -> Optional[int]:
        """
        Abstract method to perform the search.

        Returns:
        - next_state: The next state based on the specific search strategy.
        """
        pass

    def rollout(self, state: State) -> float:
        """
        Perform random simulations from the current state to terminal states.

        Args:
        - state: The current state to start the rollouts from.

        Returns:
        - avg_cost: The average total cost from the start state to the terminal states.
        """
        args = [(state, self.max_depth) for _ in range(self.num_sim)]
        with multiprocessing.Pool(processes=self.num_processes) as pool:
            results = pool.starmap(simulate_rollout, args)

        avg_cost = sum(results) / len(results)
        return avg_cost


class OneStepLookaheadWithRollout(Search):
    def search(self) -> Optional[int]:
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
        return best_action


class TwoStepLookaheadWithRollout(Search):
    def search(self) -> Optional[int]:
        best_action = None
        min_cost = float("inf")
        current_state = self.state.get_current_state()
        for action in self.state.get_actions():
            immediate_cost = self.state.get_cost(current_state, action)
            next_state = self.state.next_state(action)

            second_min_cost = float("inf")
            for second_action in next_state.get_actions():
                second_state_cost = next_state.get_cost(next_state, second_action)
                resulting_state = next_state.next_state(second_action)
                future_cost_estimate = self.rollout(resulting_state)
                total_future_cost = second_state_cost + future_cost_estimate

                if total_future_cost < second_min_cost:
                    second_min_cost = total_future_cost

            if second_min_cost == float("inf"):
                second_min_cost = 0.0
            total_cost = immediate_cost + second_min_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_action = action
        return best_action


class SelectiveDepthLookaheadWithRollout(Search):
    def __init__(
        self,
        state: State,
        num_sim: int = 100,
        max_depth: int = 1,
        should_expand_fn: Optional[Callable[[State, int], bool]] = None,
        num_processes: Optional[int] = None,
    ):
        """
        Initialize the selective depth lookahead with rollout.
        Args:
        - state: The initial state of the problem.
        - num_sim: The number of simulations to run in the rollout.
        - max_depth: The maximum depth to look ahead.
        - should_expand_fn: Optional function to determine whether to expand a node.
        """
        super().__init__(state, num_sim, num_processes)
        self.max_depth = max_depth
        self.should_expand_fn = should_expand_fn

    def search(self) -> Optional[int]:
        """
        Perform a selective depth lookahead search.
        Returns:
        - best_action: The best action determined by the search.
        """
        best_action, _ = self.recursive_search(self.state, depth=0)
        return best_action

    def recursive_search(self, state: State, depth: int) -> Tuple[Optional[int], float]:
        """
        Recursively perform the search up to the specified depth.
        Args:
        - state: The current state in the search.
        - depth: The current depth in the search tree.
        Returns:
        - best_action: The best action found at this level.
        - min_cost: The minimum estimated cost from this state.
        """
        if state.is_terminal():
            return None, state.total_cost()

        if depth >= self.max_depth:
            est_cost = self.rollout(state)
            return None, est_cost

        if self.should_expand_fn and not self.should_expand_fn(state, depth):
            est_cost = self.rollout(state)
            return None, est_cost

        best_action = None
        min_cost = float("inf")
        current_state = state.get_current_state()

        for action in state.get_actions():
            immediate_cost = state.get_cost(current_state, action)
            next_state = state.next_state(action)
            _, future_cost = self.recursive_search(next_state, depth + 1)
            total_cost = immediate_cost + future_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_action = action

        return best_action, min_cost


class MCTSNode:
    def __init__(
        self,
        state: State,
        parent: Optional[MCTSNode] = None,
        action: Optional[int] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.state.get_actions())

    def best_child(self, c_param: float = 0.1) -> MCTSNode:
        choices_weights = [
            (child.total_reward / child.visits)
            + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def tried_actions(self) -> List[Optional[int]]:
        return [child.action for child in self.children]

    def expand(self) -> MCTSNode:
        tried_actions = self.tried_actions()
        untried_actions = [
            action for action in self.state.get_actions() if action not in tried_actions
        ]
        action = random.choice(untried_actions)
        next_state = self.state.next_state(action)
        child_node = MCTSNode(state=next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node


class MonteCarloTreeSearch(Search):
    def __init__(self, state: State, num_sim: int = 1000):
        super().__init__(state, num_sim)
        self.num_sim = num_sim

    def search(self) -> Optional[int]:
        root = MCTSNode(state=self.state)
        for _ in range(self.num_sim):
            node = self.tree_policy(root)
            reward = self.base_policy(node)
            self.backpropagate(node, reward)
        best_child = root.best_child(c_param=0)
        return best_child.action

    def tree_policy(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child()
        return node

    def base_policy(self, node: MCTSNode) -> float:
        current_state = node.state
        while not current_state.is_terminal():
            actions = current_state.get_actions()
            assert actions
            action = self.rollout_policy(actions)
            current_state = current_state.next_state(action)
        return 1.0 / current_state.total_cost()

    def rollout_policy(self, possible_actions: List[int]) -> int:
        """Select an action during the rollout phase."""
        return random.choice(possible_actions)

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent  # type: ignore
