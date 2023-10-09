import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from . import CostModel

class GeneticAlgorithm:
    def __init__(
        self,
        individual_length,
        cost_model: CostModel,
        valid_allocations: list,
        population_size = 64,
        num_generations = 100,
        pop: list = None,
        prob_crossover = 0.7,
        prob_mutate = 0.3,
        hof_num = 5
        ) -> None:
        
        self.individual_length = individual_length
        self.valid_allocations = valid_allocations
        self.population_size = population_size #population of every generation
        self.para_mu = int(self.population_size / 2) #number of individuals chosen from previous generation
        self.num_generations = num_generations #number of generations
        self.cost_model = cost_model
        
        if pop is None :
            self.pop = []
        else :
            self.pop = pop #pre-set list of individuals
        
        self.prob_crossover = prob_crossover #probability of crossover
        self.prob_mutate = prob_mutate #probability of mutation

        creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
        creator.create("Individual", list, fitness = creator.FitnessMin)

        self.toolbox = base.Toolbox()
        
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.random_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", self.mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=4)

        self.toolbox.register("evaluate", self.FitnessEvaluate)

        self.hof = tools.HallOfFame(hof_num)

    def FitnessEvaluate(self, individual: list) :
        self.cost_model.set_core_allocation(individual)
        return self.cost_model.get_latency(),

    def random_individual(self) :
        return [random.choice(self.valid_allocations) for _ in range(self.individual_length)]
    
    def init_population(self):
        population = []

        for ind in self.pop:
            if len(population) >= self.population_size // 4:
                break
            population.append(ind)

        rest_num = self.population_size - len(population)
        population.extend(self.toolbox.population(n = rest_num))

        return population

    def mutate(self, individual: list) :
        if random.random() < 0.5 :
            #randomly change one allocation
            position = random.choice(range(len(individual)))
            current_allocation = individual[position]
            possible_alllocation = [x for x in self.valid_allocations if x != current_allocation]
            individual[position] = random.choice(possible_alllocation)

        else :
            #randomly exchange two allocations
            position1, position2 = random.sample(range(len(individual)), 2)
            temp = individual[position1]
            individual[position1] = individual[position2]
            individual[position2] = temp

        return individual,

    def run(self) :
        population = self.init_population()
        logbook = algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu = self.para_mu,
            lambda_ = self.population_size,
            cxpb = self.prob_crossover,
            mutpb = self.prob_mutate,
            ngen = self.num_generations,
            halloffame = self.hof
        )

        return self.hof
