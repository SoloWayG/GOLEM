from typing import Sequence, Union, Any

from golem.core.dag.graph import Graph
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.crossover import Crossover, SinglePredefinedGraphCrossover
from golem.core.optimisers.genetic.operators.elitism import Elitism
from golem.core.optimisers.genetic.operators.inheritance import Inheritance
from golem.core.optimisers.genetic.operators.mutation import Mutation, SinglePredefinedMutation, \
    SinglePredefinedGraphMutation
from golem.core.optimisers.genetic.operators.operator import PopulationT, EvaluationOperator
from golem.core.optimisers.genetic.operators.regularization import Regularization
from golem.core.optimisers.genetic.operators.reproduction import ReproductionController
from golem.core.optimisers.genetic.operators.selection import Selection
from golem.core.optimisers.genetic.parameters.graph_depth import AdaptiveGraphDepth
from golem.core.optimisers.genetic.parameters.operators_prob import init_adaptive_operators_prob
from golem.core.optimisers.genetic.parameters.population_size import init_adaptive_pop_size, PopulationSize
from golem.core.optimisers.objective.objective import Objective
from golem.core.optimisers.opt_history_objects.individual import Individual
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams
from golem.core.optimisers.populational_optimizer import PopulationalOptimizer


class EvoGraphOptimizer(PopulationalOptimizer):
    """
    Multi-objective evolutionary graph optimizer named GPComp
    """

    def __init__(self,
                 objective: Objective,
                 initial_graphs: Sequence[Union[Graph, Any]],
                 requirements: GraphRequirements,
                 graph_generation_params: GraphGenerationParams,
                 graph_optimizer_params: GPAlgorithmParameters):
        super().__init__(objective, initial_graphs, requirements, graph_generation_params, graph_optimizer_params)
        # Define genetic operators
        self.regularization = Regularization(graph_optimizer_params, graph_generation_params)
        self.selection = Selection(graph_optimizer_params)
        self.crossover = SinglePredefinedGraphCrossover(graph_optimizer_params, requirements, graph_generation_params)
        self.mutation = SinglePredefinedGraphMutation(graph_optimizer_params, requirements, graph_generation_params)
        self.inheritance = Inheritance(graph_optimizer_params, self.selection)
        self.elitism = Elitism(graph_optimizer_params)
        self.operators = [self.regularization, self.selection, self.crossover,
                          self.mutation, self.inheritance, self.elitism]

        self.reproducer = ReproductionController(parameters=graph_optimizer_params,
                                                 selection=self.selection,
                                                 mutation=self.mutation,
                                                 crossover=self.crossover,
                                                 verifier=self.graph_generation_params.verifier)

        # Define adaptive parameters
        self._pop_size: PopulationSize = init_adaptive_pop_size(graph_optimizer_params, self.generations)
        self._operators_prob = init_adaptive_operators_prob(graph_optimizer_params)
        self._graph_depth = AdaptiveGraphDepth(self.generations,
                                               start_depth=requirements.start_depth,
                                               max_depth=requirements.max_depth,
                                               max_stagnation_gens=graph_optimizer_params.adaptive_depth_max_stagnation,
                                               adaptive=graph_optimizer_params.adaptive_depth)

        # Define initial parameters
        self.requirements.max_depth = self._graph_depth.initial
        self.graph_optimizer_params.pop_size = self._pop_size.initial
        self.initial_individuals = [Individual(graph, metadata=requirements.static_individual_metadata)
                                    for graph in self.initial_graphs]

    def _initial_population(self, evaluator: EvaluationOperator):
        """ Initializes the initial population """
        # Adding of initial assumptions to history as zero generation
        self._update_population(evaluator(self.initial_individuals), 'initial_assumptions')
        # pop_size = self.graph_optimizer_params.pop_size
        #
        # if len(self.initial_individuals) < pop_size:
        #     self.initial_individuals += self.reproducer._mutate_over_population(population=self.initial_individuals,
        #                                                                         evaluator=evaluator)
        #     # Adding of extended population to history
        #     self._update_population(self.initial_individuals, 'extended_initial_assumptions')

    def _evolve_population(self, evaluator: EvaluationOperator) -> PopulationT:
        """ Method realizing full evolution cycle """

        # Defines adaptive changes to algorithm parameters
        #  like pop_size and operator probabilities
        self._update_requirements()

        # Regularize previous population
        individuals_to_select = self.regularization(self.population, evaluator)
        # Reproduce from previous pop to get next population
        new_population = self.reproducer.reproduce(individuals_to_select, evaluator)

        # Adaptive agent experience collection & learning
        # Must be called after reproduction (that collects the new experience)
        experience = self.mutation.agent_experience
        experience.collect_results(new_population)
        self.mutation.agent.partial_fit(experience)

        # Use some part of previous pop in the next pop
        new_population = self.inheritance(self.population, new_population)
        new_population = self.elitism(self.generations.best_individuals, new_population)
        return new_population

    def _update_requirements(self):
        if not self.generations.is_any_improved:
            self.graph_optimizer_params.mutation_prob, self.graph_optimizer_params.crossover_prob = \
                self._operators_prob.next(self.population)
            self.log.info(
                f'Next mutation proba: {self.graph_optimizer_params.mutation_prob}; '
                f'Next crossover proba: {self.graph_optimizer_params.crossover_prob}')
        self.graph_optimizer_params.pop_size = self._pop_size.next(self.population)
        self.requirements.max_depth = self._graph_depth.next()
        self.log.info(
            f'Next population size: {self.graph_optimizer_params.pop_size}; '
            f'max graph depth: {self.requirements.max_depth}')

        # update requirements in operators
        for operator in self.operators:
            operator.update_requirements(self.graph_optimizer_params, self.requirements)
