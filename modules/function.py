import numpy as np

class Function:
    """
    Classe para representar uma função matemática.
    """
    def __init__(self, func, bounds=None):
        """
        Inicializa a função com os parâmetros fornecidos.

        :param func: Função a ser representada
        :param bounds: Limites inferior e superior da função
        """
        self.func = func
        self.bounds = bounds
        if bounds is not None:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = None
            self.upper_bound = None

    def __call__(self, x):
        """
        Chama a função com o valor fornecido.

        :param x: Valor de entrada
        :return: Valor da função
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            if np.any(x < self.lower_bound) or np.any(x > self.upper_bound):
                raise ValueError("Valor fora dos limites definidos.")
        return self.func(x)

    def fitness(self, x):
        """
        Calcula a função de adequação.

        :param x: Array de entrada
        :return: Valor da função de adequação
        """
        fx = self.func(x)
        return 100 / (1 + fx)
