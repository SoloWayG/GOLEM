import os
import time
import glob

from datetime import timedelta
from examples.synthetic_graph_evolution.generators import generate_labeled_graph
from golem.core.adapter.nx_adapter import BaseNetworkxAdapter
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.objective import Objective
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.paths import default_data_dir
from golem.metrics.edit_distance import tree_edit_dist
from functools import partial


def find_latest_file_in_dir(directory: str) -> str:
    return max(glob.glob(os.path.join(directory, '*')), key=os.path.getmtime)

def test_saved_state():
    # Set params
    size = 16
    early_stopping_iterations = 200
    num_of_generations_run_1 = 17  # 40 100 10
    num_of_generations_run_2 = 19  # 45 120 12
    timeout = 10
    saved_state_path = 'saved_optimisation_state/test'

    # Generate target graph sought by optimizer using edit distance objective
    node_types = ('a', 'b')  # Available node types that can appear in graphs
    target_graph = generate_labeled_graph('tree', size, node_types)
    objective = Objective(partial(tree_edit_dist, target_graph))
    initial_population = [generate_labeled_graph('tree', 5, node_types) for _ in range(10)]

    # Setup optimization parameters
    requirements_run_1 = GraphRequirements(timeout=timedelta(minutes=timeout),
                                     early_stopping_iterations=early_stopping_iterations,
                                     num_of_generations=num_of_generations_run_1)
    requirements_run_2 = GraphRequirements(timeout=timedelta(minutes=timeout),
                                           early_stopping_iterations=early_stopping_iterations,
                                           num_of_generations=num_of_generations_run_2)

    gen_params = GraphGenerationParams(adapter=BaseNetworkxAdapter(), available_node_types=node_types)
    algo_params = GPAlgorithmParameters(pop_size=30)

    # Build and run the optimizer to create a saved state file
    optimiser1 = EvoGraphOptimizer(objective, initial_population, requirements_run_1, gen_params, algo_params,
                                  saved_state_path=saved_state_path)
    st = time.time()
    optimiser1.optimise(objective)
    et = time.time()
    time1 = int(et - st) / 60

    # Check that the file with saved state was created
    saved_state_full_path = os.path.join(default_data_dir(), saved_state_path, optimiser1._run_id)
    saved_state_file = find_latest_file_in_dir(saved_state_full_path)
    assert os.path.isfile(saved_state_file) is True, 'ERROR: Saved state file was not created!'

    # Build and run the optimizer to check that the saved state was used
    optimiser2 = EvoGraphOptimizer(objective, initial_population, requirements_run_2, gen_params, algo_params,
                                  use_saved_state=True, saved_state_path=saved_state_path)  # saved_state_file='/private/tmp/GOLEM/saved_optimisation_state/test/f26199f8-b947-11ee-8ad4-1e00f32e993a/1705943286.pkl'

    # Check that the restored object has the same main parameters as the original
    assert optimiser1.current_generation_num == optimiser2.current_generation_num
    assert optimiser1.generations.stagnation_iter_count == optimiser2.generations.stagnation_iter_count
    assert optimiser1.best_individuals == optimiser2.best_individuals
    assert optimiser1.population == optimiser2.population

    st = time.time()
    optimiser2.optimise(objective)
    et = time.time()
    time2 = int(et - st) / 60

    print(time1)
    print(time2)

    # Check that the run with the saved state takes less time than it would without it
    # assert time1 > 2
    # assert time2 < 0.5

    # Check that the result of the second optimisation is the same as or better than the first
    assert optimiser2.best_individuals[0].fitness.value >= optimiser1.best_individuals[0].fitness.value
