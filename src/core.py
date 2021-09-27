from math import log2

class VariableChanger(object):
    """
    Variable changer for quadratic forms class
    """
    
    def __init__(self, B, t, matrix_factory, k=None, gamma=1):
        """
        Arguments:\n
        B -- lattice basis\n
        t -- target vector\n
        matrix_factory -- class for matrix generation\n
        k -- qubits per coordinate (default is None. In this case k is computed automaticly)\n
        gamma -- BDD parametr (default is 1)
        """
        self.B = B
        self.t = t
        self.matrix_factory = matrix_factory
        self.n = B.shape[0]
        
        if k is None:
            self.k = int(log2(2 * gamma * self.n**(1/2) * 
                abs(B.det())**(1/self.n) + 
                sum(abs(t[i]) for i in range(self.n))))
        else:
            self.k = k
            
        self.T = self.__create_transition_matrix()
        self.d = 2**self.k * self.matrix_factory.getOnes(self.n, 1)
        
    def __create_transition_matrix(self):
        data = [[0 for _ in range((self.k + 1)*self.n)] for _ in range(self.n)]
        T = self.matrix_factory.getMatrix(data)

        for i in range(self.n):
            T[i, i*(self.k+1):i*(self.k+1)+self.k+1] = \
                self.matrix_factory.getMatrix([[2**j for j in range(self.k+1)]])
        
        return T
    
    def change_quadratic_form(self) -> tuple:
        """
        Computing quadratic form's coefficients in QUBO variables
        """
        quadratic_part = self.T.T @ self.B.T @ self.B @ self.T

        linear_part = -2 * self.d.T @ self.B.T @ self.B @ self.T - \
            2 * self.t.T @ self.B @ self.T

        const = self.d.T @ self.B.T @ self.B @ self.d + \
            2 * self.d.T @ self.B.T @ self.t + self.t.T @ self.t
        
        return quadratic_part, linear_part, const[0]
    
    def straight_change_variables(self, x):
        """
        Going from integer variables to QUBO ones.\n
        Arguments:\n
        x -- vector in integer variables. \n
        straight_change_variables(x) -> Vector in QUBO variables
        """
        res = self.matrix_factory.getZeros(self.n*(self.k + 1), 1)
        
        for i in range(len(x)):
            bin_repr = list(bin(x[i] + 2**self.k)[2:].zfill(self.k+1)[::-1])
            bin_repr = list(map(lambda el: [int(el)], bin_repr))
            res[(self.k+1)*i:(self.k+1)*(i+1), 0] = bin_repr
        
        return res
    
    def backward_change_variables(self, x):
        """
        Going from QUBO variables to integer ones.\n
        Arguments:\n
        x -- vector in QUBO variables.\n
        backward_change_variables(x) -> Vector in integer variables
        """

        return self.T@x - self.d

class VectorSet(object):
    """
    Record for result of quantum annealing.\n
    prob -- probability of given vector\n
    norm -- norm of given vector\n
    vectors -- list of cectors with the same norm
    """
    
    def __init__(self, norm : int, vectors : list, prob : float):
        self.norm = norm
        self.vectors = vectors
        self.prob = prob

    def __str__(self) -> str:
        return f"(norm: {self.norm}, prob: {self.prob}, vectors: {list(map(lambda x: list(x), self.vectors))})"
        
class Solver(object):
    """
    QUBO solver class
    """
    
    def __init__(self, B, t, matrix_factory, sampler, k=None, gamma=1):
        """
        Arguments:\n
        B -- lattice basis\n
        t -- target vector\n
        matrix_factory -- class for matrix generation\n
        sampler - annealing sampler
        k -- qubits per coordinate (default is None)\n
        gamma -- BDD parametr (default is 1)
        """
        self.B, self.t = B, t
        self.matrix_factory = matrix_factory
        self.var_changer = VariableChanger(B, t, self.matrix_factory, k, gamma)
        self.sampler = sampler
        
    def __norm(self, x):
        return ((self.B@x - self.t).T @ (self.B@x - self.t))[0]
        
    def __generate_qubo_input(self, quadratic, linear):
        Q = {}
    
        for i in range(quadratic.shape[0]):
            for j in range(i, quadratic.shape[1]):
                tmp = (f"x_{i}", f"x_{j}")
                Q[tmp] = int(quadratic[i, j] + linear[i]) if i == j else int(2*quadratic[i, j])
            
        return Q

    def __dict_to_vector(self, vec_repr):
        vec = self.matrix_factory.getZeros(len(vec_repr), 1)
        
        for record in list(vec_repr.items()):
            vec[int(record[0][2:])] = record[1]
            
        return vec
    
    def __interpret_qubo_output(self, sampleset, const, **params) -> list:
        res = {}
    
        for sample in sampleset.data():
            x = self.var_changer.backward_change_variables(self.__dict_to_vector(sample.sample))
            val = self.__norm(x)
        
            if val in res:
                res[val][0].append(x)
                res[val][1] += sample.num_occurrences/params["num_reads"]
            else:
                res[val] = [[x], sample.num_occurrences/params["num_reads"]]
        
        return list(map(lambda x: VectorSet(x[0], x[1][0], x[1][1]), res.items()))
    
    def solve(self, **params) -> list[VectorSet]:
        """
        Solving QUBO problem on annealing sampler.\n
        Params:\n
        annealing_time -- annealing time per sample in ms\n
        num_reads -- number of sweaps on sampler per call
        """
        q, l, c = self.var_changer.change_quadratic_form()
        Q = self.__generate_qubo_input(q, l)
        sampleset = self.sampler.sample_qubo(Q, **params)
        
        return self.__interpret_qubo_output(sampleset, c, **params)

class MatrixFactory(object):
    """
    Class for matrix generation
    """
    def getMatrix(self, data : list[list[int]]):
        pass

    def getRandomMatrix(self, dim : int, bit_length : int):
        pass

    def fromString(self, data : str):
        pass
    
    def getOnes(self, n : int, k : int):
        pass
    
    def getZeros(self, n : int, k : int):
        pass
    
    def getEye(self, n : int):
        pass