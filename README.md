This is a search algorithm that performs a grid search for the final location with a visualized output.

I implemented multiple algorithms:
dfs
bfs
gbfs
as (a star)
dijkstra
idas (iterative deepening a star)
AND heuristics :
"manhattan", "euclidean", "chebyshev"
just download and test run the provided test cases(.txt)
Navigate in the terminal to the repository and use the command:
usage form: python main.py <filename> <algorithm> [heuristic]
for example Example: python main.py test_cases/test.txt dfs
or options of algorithms that use heuristic functions
for example: Example: python main.py test_cases/test.txt as manhattan"
Don't worry if you do not know what uses a heuristic function or not. I wrote the code to use Manhattan  if no arguments are given by default.
