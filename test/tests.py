import unittest
import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from factories import *
from core import *
from interactors import *
from controllers import *
from dao import *

from neal import SimulatedAnnealingSampler
from dwave.system import DWaveSampler, EmbeddingComposite

class TestClass(unittest.TestCase):

    def setUp(self) -> None:
        self.token = JsonAuthController("resources/auth.json").get_token()

    def get_matricies(self, ind):
        dao = MatrixStorage("resources/matrix_storage.csv", SympyMatrixFactory())
        B = dao.read(ind, "Matrix")
        t = dao.read(ind, "Vector")

        return B, t

    def get_matricies_list(self, size):
        res = []
        for i in range(size):
            res.append(self.get_matricies(i))

        return res

    def test_generate_samples(self):
        dim = 3
        bit_count = 2
        small_factory = SmallMatrixFactory(SympyMatrixFactory())
        dao = MatrixStorage("resources/matrix_storage.csv", small_factory)

        for i in range(20):
            B = small_factory.getRandomMatrix(dim, bit_count)
            t = small_factory.getZeros(dim, 1)
            dao.write(B, t, f"Small matrix sample {i}")

    def test_clean_db(self):
        dao = MatrixStorage("resources/matrix_storage.csv")
        dao.clean()

    def test_simulated_stat(self):
        B, t = self.get_matricies(18)
        sampler = SimulatedAnnealingSampler()
        interactor = TestInteractor(B, t, SympyMatrixFactory(), sampler)
        interactor.compute_probabilities((2, 10), num_reads=1000)
        interactor.show_plot(show=False, image_name="resources/energy_levels_simulated.png")

    def test_quantum_stat(self):
        B, t = self.get_matricies(18)
        sampler = EmbeddingComposite(DWaveSampler(token=self.token, 
            solver={'topology__type': 'chimera'}))
        interactor = TestInteractor(B, t, SympyMatrixFactory(), sampler)
        interactor.compute_probabilities((2, 8), num_reads=1000, k=2)
        interactor.show_plot(show=False, image_name="resources/energy_levels_quantum_18.png")

    def test_average_simulated(self):
        inp = self.get_matricies_list(20)
        sampler = SimulatedAnnealingSampler()
        interactor = StatCollectingInteractor(inp, SympyMatrixFactory(), sampler)
        interactor.compute_probabilities((2, 10), num_reads=1000)
        interactor.show_plot(show=False, image_name="resources/average_simulated.png")

    def test_average_quantum(self):
        inp = self.get_matricies_list(20)
        sampler = EmbeddingComposite(DWaveSampler(token=self.token, 
            solver={'topology__type': 'chimera'}))        
        interactor = StatCollectingInteractor(inp, SympyMatrixFactory(), sampler)
        interactor.compute_probabilities((2, 8), num_reads=100, k=2)
        interactor.show_plot(show=False, image_name="resources/average_quantum.png")

if __name__ == "__main__":
    logger = logging.getLogger("diploma")
    handler = logging.FileHandler("resources/test.log", mode="a")
    handler.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s\n%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info("***NEW TEST SESSION IS STARTING***")

    suite = unittest.TestSuite()
    #suite.addTest(TestClass("test_generate_samples"))
    #suite.addTest(TestClass("test_clean_db"))
    #suite.addTest(TestClass("test_simulated_stat"))
    #suite.addTest(TestClass("test_quantum_stat"))
    suite.addTest(TestClass("test_average_simulated"))
    #suite.addTest(TestClass("test_average_quantum"))
    runner = unittest.TextTestRunner()
    runner.run(suite)