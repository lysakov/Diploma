import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from factories import *
from core import *

from sympy import Matrix
from random import randint

class CoreTests(unittest.TestCase):
    
    def setUp(self) -> None:
        self.n = 5
        self.N = 16
        self.B = Matrix([[randint(-2**(self.N-1), 2**(self.N-1)-1) for _ in range(self.n)] 
            for _ in range(self.n)], dtype=object)
        self.t = Matrix([[randint(-2**(self.N-1), 2**(self.N-1)-1)] for _ in range(self.n)])
        self.var_changer = VariableChanger(self.B, self.t, SympyMatrixFactory())

    def compute_form(self, Q, L, C, x):
        return (x.T@Q@x)[0] + (L@x)[0] + C

    def test_variable_changer(self):
        x = Matrix([[randint(-2**(self.N-1), 2**(self.N-1)-1)] for _ in range(self.n)])
        x1 = self.var_changer.straight_change_variables(x)
        self.assertEqual(x, self.var_changer.backward_change_variables(x1))

    def test_quadratic_form_changer(self):
        x = Matrix([[randint(-2**(self.N-1), 2**(self.N-1)-1)] for _ in range(self.n)])
        x1 = self.var_changer.straight_change_variables(x)
        q, l, c = self.var_changer.change_quadratic_form()
        norm1 = self.compute_form(self.B.T@self.B, -2*self.t.T@self.B, (self.t.T@self.t)[0], x)
        norm2 = self.compute_form(q, l, c, x1)
        norm3 = (self.B@x - self.t).dot(self.B@x - self.t)
        self.assertEqual(norm1, norm2)
        self.assertEqual(norm2, norm3)

if __name__ == "__main__":
    unittest.main()