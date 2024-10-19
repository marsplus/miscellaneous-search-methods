# type: ignore
import random
from itertools import permutations

import pytest

from app.search import (
    MonteCarloTreeSearch,
    OneStepLookaheadWithRollout,
    SelectiveDepthLookaheadWithRollout,
    TwoStepLookaheadWithRollout,
)
from app.tsp import TSPState

N = 10000
N_PROC = 20

random.seed(42)


def should_expand(state, depth):
    return True


def calculate_tour_cost(tour, dist_matrix):
    """Calculate the total cost of a given tour."""
    total_cost = sum(dist_matrix[tour[i - 1]][tour[i]] for i in range(1, len(tour)))
    total_cost += dist_matrix[tour[-1]][tour[0]]
    return total_cost


def find_optimal_tsp_solution(dist_matrix):
    """Find the optimal TSP solution using brute-force."""
    cities = list(range(len(dist_matrix)))
    best_tour, min_cost = min(
        (
            (tour, calculate_tour_cost(tour, dist_matrix))
            for tour in permutations(cities)
        ),
        key=lambda x: x[1],
    )
    return best_tour, min_cost


@pytest.fixture
def small_distance_matrix():
    return [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]


@pytest.fixture
def ten_city_distance_matrix():
    return [
        [0, 15, 95, 75, 31, 10, 88, 85, 20, 25],
        [15, 0, 30, 80, 40, 65, 91, 15, 95, 70],
        [95, 30, 0, 69, 77, 11, 84, 88, 75, 54],
        [75, 80, 69, 0, 14, 20, 62, 99, 28, 89],
        [31, 40, 77, 14, 0, 17, 26, 72, 93, 92],
        [10, 65, 11, 20, 17, 0, 67, 82, 97, 18],
        [88, 91, 84, 62, 26, 67, 0, 41, 15, 51],
        [85, 15, 88, 99, 72, 82, 41, 0, 18, 36],
        [20, 95, 75, 28, 93, 97, 15, 18, 0, 62],
        [25, 70, 54, 89, 92, 18, 51, 36, 62, 0],
    ]


def run_search_method(search_method, initial_state, optimal_cost):
    """Run the search method for the TSP and verify the total cost matches the optimal cost."""
    if isinstance(search_method, MonteCarloTreeSearch):
        search_method.num_sim = N * 10
    else:
        search_method.num_sim = N
    search_method.num_processes = N_PROC

    current_state = initial_state
    while not current_state.is_terminal():
        next_city = search_method.search()
        current_state = current_state.next_state(next_city)
        search_method.update_state(current_state)
    total_cost = current_state.total_cost()
    acceptable_cost = optimal_cost * 1.05  # Allow a 5% tolerance
    assert total_cost <= acceptable_cost


def run_single_step_search_test(search_class, distance_matrix):
    """Run a search method for one-step or two-step lookahead and verify the solution."""
    tour, optimal_cost = find_optimal_tsp_solution(distance_matrix)
    start, second = tour[0], tour[1]
    initial_state = TSPState(
        visited=[start], city=start, dist=distance_matrix, num_city=len(distance_matrix)
    )

    search_method = search_class(initial_state, num_sim=N)

    # Run the search method and verify the total cost
    run_search_method(search_method, initial_state, optimal_cost)


def run_selective_lookahead_test(distance_matrix, max_depth):
    """Run a selective depth lookahead search and verify the solution."""
    tour, optimal_cost = find_optimal_tsp_solution(distance_matrix)
    start, second = tour[0], tour[1]
    initial_state = TSPState(
        visited=[start], city=start, dist=distance_matrix, num_city=len(distance_matrix)
    )

    selective_search = SelectiveDepthLookaheadWithRollout(
        state=initial_state,
        num_sim=N,
        max_depth=max_depth,
        should_expand_fn=should_expand,
    )

    # When max_depth=1 or 2, compare with the corresponding lookahead
    if max_depth == 1:
        compare_search = OneStepLookaheadWithRollout(initial_state, num_sim=N)
    elif max_depth == 2:
        compare_search = TwoStepLookaheadWithRollout(initial_state, num_sim=N)
    else:
        compare_search = None

    if compare_search is not None:
        # Increase num_sim in the search methods if needed, due to the randomness of rollout
        selective_search.num_sim = N
        compare_search.num_sim = N

        # Test if the first action is the same
        next_city_selective = selective_search.search()
        next_city_compare = compare_search.search()
        assert next_city_selective == next_city_compare

    # Run the search method and verify the total cost
    run_search_method(selective_search, initial_state, optimal_cost)


def run_mcts_search_test(search_class, distance_matrix):
    """Run MCTS search and verify the solution."""
    tour, optimal_cost = find_optimal_tsp_solution(distance_matrix)
    start = tour[0]
    initial_state = TSPState(
        visited=[start], city=start, dist=distance_matrix, num_city=len(distance_matrix)
    )

    search_method = search_class(state=initial_state, num_sim=N)
    run_search_method(search_method, initial_state, optimal_cost)


# Test One-Step Lookahead on a small matrix
def test_one_step_lookahead_small(small_distance_matrix):
    run_single_step_search_test(OneStepLookaheadWithRollout, small_distance_matrix)


# Test Two-Step Lookahead on a small matrix
def test_two_step_lookahead_small(small_distance_matrix):
    run_single_step_search_test(TwoStepLookaheadWithRollout, small_distance_matrix)


# Test One-Step Lookahead on a 10-city matrix
def test_one_step_lookahead_ten_city(ten_city_distance_matrix):
    run_single_step_search_test(OneStepLookaheadWithRollout, ten_city_distance_matrix)


# Test Two-Step Lookahead on a 10-city matrix
def test_two_step_lookahead_ten_city(ten_city_distance_matrix):
    run_single_step_search_test(TwoStepLookaheadWithRollout, ten_city_distance_matrix)


# Test Selective Lookahead with Depth 1 on a small matrix
def test_selective_lookahead_depth_one_small(small_distance_matrix):
    run_selective_lookahead_test(small_distance_matrix, max_depth=1)


# Test Selective Lookahead with Depth 2 on a small matrix
def test_selective_lookahead_depth_two_small(small_distance_matrix):
    run_selective_lookahead_test(small_distance_matrix, max_depth=2)


# Test Selective Lookahead with Depth 1 on a 10-city matrix
def test_selective_lookahead_depth_one_ten_city(ten_city_distance_matrix):
    run_selective_lookahead_test(ten_city_distance_matrix, max_depth=1)


# Test Selective Lookahead with Depth 2 on a 10-city matrix
def test_selective_lookahead_depth_two_ten_city(ten_city_distance_matrix):
    run_selective_lookahead_test(ten_city_distance_matrix, max_depth=2)


# Test MCTS on a small matrix
def test_mcts_small(small_distance_matrix):
    run_mcts_search_test(MonteCarloTreeSearch, small_distance_matrix)


# Test MCTS on a 10-city matrix
def test_mcts_ten_city(ten_city_distance_matrix):
    run_mcts_search_test(MonteCarloTreeSearch, ten_city_distance_matrix)


# Test terminal state handling
def test_terminal_state(small_distance_matrix):
    initial_state = TSPState(
        visited=[0],
        city=0,
        dist=small_distance_matrix,
        num_city=len(small_distance_matrix),
    )
    terminal_state = initial_state.next_state(1).next_state(2).next_state(3)
    assert terminal_state.is_terminal() == True  # noqa: E712
    assert terminal_state.get_actions() == []
