import numpy as np
import matplotlib.pyplot as plt
from .function import Function

class Genetics:
    """
    Classe para representar uma população de indivíduos.
    """

    def __init__(self, size, dim, upper_bound=5, lower_bound=-5, function=None):
        """
        Inicializa a população com os parâmetros fornecidos.

        :param size: Tamanho da população
        :param dim: Dimensão de cada indivíduo
        :param upper_bound: Limite superior de cada dimensão
        :param lower_bound: Limite inferior de cada dimensão
        """
        self.size = size
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.function = function
        if function is None:
            raise ValueError("Função não definida.")
        if size <= 0 or dim <= 0:
            raise ValueError("Tamanho e dimensão devem ser maiores que zero.")
        if upper_bound <= lower_bound:
            raise ValueError("Limite superior deve ser maior que o limite inferior.")
        if not isinstance(function, Function):
            raise ValueError("Função deve ser uma instância da classe Function.")
        self.individuals = self.generate_population(size, dim, upper_bound, lower_bound)


    def __call__(self, max_generations=1000, k=2, mutation_probability = 0.1):
        """
        Executa o algoritmo genético.

        :param max_generations: Número máximo de gerações
        :param k: Número de indivíduos a serem selecionados para reprodução
        :return: Melhor indivíduo encontrado e o valor da função nesse ponto
        """

        fitness = self.calculate_fitness()
        
        best_individual = self.individuals[np.argmax(fitness)]
        best_value = [self.function(best_individual)]


        for i in range(max_generations):
            # Seleção
            parents = self.rws(k=k)
            parents = np.vstack([parents, best_individual])

            # Crossover
            self.crossover(self.size, pop_aux=parents)

            # Mutação
            self.mutate(probability=mutation_probability)

            # Avaliação
            fitness = self.calculate_fitness()
            idx_best = np.argmax(fitness)
            current_best = self.function(self.individuals[idx_best])
            best_value.append(min(best_value[-1], current_best))
            if current_best < best_value[-1]:
                best_individual = self.individuals[idx_best]
            
        return best_individual, best_value


    def generate_population(self, size, dim, upper_bound=5, lower_bound=-5):
        """
        Gera uma população inicial aleatória.

        :param size: Tamanho da população
        :param dim: Dimensão de cada indivíduo
        :param upper_bound: Limite superior de cada dimensão
        :param lower_bound: Limite inferior de cada dimensão
        :return: População inicial
        """
        return np.random.uniform(low=lower_bound, high=upper_bound, size=(size, dim))


    def calculate_fitness(self):
        """
        Calcula a função de adequação dos indivíduos.

        :return: Vetor de fitness dos indivíduos
        """
        fitness = []
        for individual in self.individuals:
            ind_fitness = self.function.fitness(individual)
            fitness.append(ind_fitness)

        return np.array(fitness)


    def rws(self, k=2):
        """
        Seleção de indivíduos via método da roleta

        :param k: Quantidade de indivíduos que serão selecionados
        :return: Mantém apenas os indivíduos selecionados
        """
        fitness = self.calculate_fitness()
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            return self.individuals[np.random.choice(self.individuals.shape[0], size=k)]
        probabilities = fitness / total_fitness
        return self.individuals[np.random.choice(self.individuals.shape[0], size=k, p=probabilities)]
    

    def crossover(self, target, pop_aux):
        """
        Cria novos individuos com o cruzamento dos indivíduos dados e substitui a população atual
        
        :param target: Quantidade alvo de indivíduos
        :param population: Indivíduos iniciais para cruzamento
        """
        offspring = []
        for _ in range(target):
            parents_index = np.random.choice(len(pop_aux), 2)
            parents = pop_aux[parents_index]
            pointer = np.random.randint(1, self.dim - 1)
            child = (np.concatenate([parents[0][:pointer], parents[1][pointer:]]))
            offspring.append(child)
        self.individuals = np.vstack([pop_aux, offspring])


    def mutate(self, probability):
        """
        Aplica uma mutação aos indivíduos da população atual
        """
        for ind in (self.individuals):
            chance = np.random.uniform()
            if chance < probability:
                target = np.random.randint(0, self.dim)
                inc = np.random.randint(0, 1 + 1)
                i = np.random.randint(1, 10 + 1)
                if inc:
                    delta = (self.upper_bound - ind[target]) / 10
                    ind[target] = np.random.uniform(ind[target], ind[target] + i * delta)
                else:
                    delta = (ind[target] - self.lower_bound) / 10
                    ind[target] = np.random.uniform(ind[target] - i * delta, ind[target])