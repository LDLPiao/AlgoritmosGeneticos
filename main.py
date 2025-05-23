import numpy as np
import matplotlib.pyplot as plt
from modules.function import Function
from modules.genetics import Genetics


if __name__ == '__main__':
    func = Function(lambda x: np.sum(x**2), bounds=(-5, 5))

    size = 100
    k = int(size / 2)

    max_gen = 100
    mutation_prob = 0.1

    population = Genetics(size=size, dim=10, upper_bound=5, lower_bound=-5, function=func)
    best, value = population(max_generations=max_gen, k=k, mutation_probability=mutation_prob)
    print("Melhor indivíduo:", best)
    print("Valor da função:", float(value[-1])) 

    gen = range(len(value))

    plt.plot(gen, value)
    plt.show()