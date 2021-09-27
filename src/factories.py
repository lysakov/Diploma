from sympy import Matrix, zeros, ones, eye
from sympy.parsing.sympy_parser import parse_expr
from random import randint
from core import MatrixFactory

class SympyMatrixFactory(MatrixFactory):
    
    def getMatrix(self, data : list[list[int]]) -> Matrix:
        return Matrix(data)

    def getRandomMatrix(self, dim : int, bit_length : int) -> Matrix:
        return self.getMatrix([[randint(-2**bit_length, 2**bit_length - 1) 
            for _ in range(dim)] for _ in range(dim)])

    def fromString(self, data : str) -> Matrix:
        return parse_expr(data)
    
    def getOnes(self, n : int, k : int) -> Matrix:
        return ones(n, k)
    
    def getZeros(self, n : int, k : int) -> Matrix:
        return zeros(n, k)
    
    def getEye(self, n : int) -> Matrix:
        return eye(n)

class HNFFactory(MatrixFactory):
    
    def __init__(self, matrix_factory=SympyMatrixFactory()):
        self.matrix_factory = matrix_factory

    def getMatrix(self, data : list[list[int]]) -> Matrix:
        return self.matrix_factory.getMatrix(data)

    def getRandomMatrix(self, dim : int, bit_length : int) -> Matrix:
        B = self.matrix_factory.getEye(dim)
        k = dim - 1
        bound = 2**bit_length - 1
        
        while k >= 0:
            B[k, dim - 1] = randint(2**(bit_length-1), bound)
            bound = B[k, dim - 1]
            k -= 1
            
        return B.T

    def fromString(self, data : str) -> Matrix:
        return self.matrix_factory.fromString(data)
    
    def getOnes(self, n : int, k : int) -> Matrix:
        return self.matrix_factory.getOnes(n, k)
    
    def getZeros(self, n : int, k : int) -> Matrix:
        return self.matrix_factory.getZeros(n, k)
    
    def getEye(self, n : int) -> Matrix:
        return self.matrix_factory.getEye(n)

class SmallMatrixFactory(MatrixFactory):

    def __init__(self, matrix_factory : MatrixFactory):
        self.matrix_factory = matrix_factory

    def getMatrix(self, data: list[list[int]]):
        return self.matrix_factory.getMatrix(data)

    def getRandomMatrix(self, dim: int, bit_length: int):
        data = [[randint(0, 1) for _ in range(dim)] for _ in range(dim)]
        good_basis = self.matrix_factory.getMatrix(data)
        while good_basis.det() == 0:
            data = [[randint(0, 1) for _ in range(dim)] for _ in range(dim)]
            good_basis = self.matrix_factory.getMatrix(data)

        unimodular_matrix = self.matrix_factory.getZeros(dim, dim)
        while abs(unimodular_matrix.det()) != 1:
            unimodular_matrix = self.matrix_factory.getRandomMatrix(dim, bit_length)

        return good_basis@unimodular_matrix

    def fromString(self, data : str):
        return self.matrix_factory.fromString(data)
    
    def getOnes(self, n : int, k : int):
        return self.matrix_factory.getOnes(n, k)
    
    def getZeros(self, n : int, k : int):
        return self.matrix_factory.getZeros(n, k)
    
    def getEye(self, n : int):
        return self.matrix_factory.getEye(n)