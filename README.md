Experiment with heuristic search methods for solving problems that can be formulated as Markov Decision Processes (MDPs). 

Given a problem, the thinking process typically follows these steps: 
1. Formulate the problem as an MDP—a sequential decision-making problem.
2. Solving the MDP = Use dynamic programming (DP) to solve the Bellman optimality equation.
3. There are two ways to approximately solve the Bellman optimality equation:
    1. Approximate in the value space. 
    2. Approximate in the policy space. 
4. All the search methods below approximate in the value space.

#### Heuristic Search
- One-step lookahead with rollout
- Two-step lookahead with rollout: 
- Selective-depth lookahead with rollout
- Monte Carlo tree search (MCTS)
- Depth-first search (DFS)
- Backtracking search (a.k.a DFS with pruning)


#### Testbed Problems
- Traveling Salesman Problem (TSP)


