from app.search import OneStepLookaheadWithRollout
from app.tsp import TSPState
from app.utils import generate_distance_matrix


def main() -> None:
    N = 20
    dist_matrix = generate_distance_matrix(N)
    initial_state = TSPState(visited=[0], city=0, dist=dist_matrix, num_city=N)
    one_step_search = OneStepLookaheadWithRollout(initial_state)
    current_state = initial_state
    while not current_state.is_terminal():
        next_city = one_step_search.search()
        print(f"Next city to visit: {next_city}")
        current_state = current_state.next_state(next_city)
        one_step_search.update_state(current_state)
    print(f"Total cost of the tour: {current_state.total_cost()}")


if __name__ == "__main__":
    main()
