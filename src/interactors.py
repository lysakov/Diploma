import matplotlib.pyplot as plt
import logging

from numpy import array, average
from numpy.core.shape_base import vstack

from core import *

class TestInteractor(object):
    """
    Class computes probabilities of ground and first two energy states
    for BDD problem with gamma=1
    """
    
    def __init__(self, B, t, matrix_factory, sampler):
        """
        Arguments:\n
        B -- lattice basis\n
        t -- target vector\n
        matrix_factory -- class for matrix generation\n
        sampler - annealing sampler
        """
        self.B = B
        self.t = t
        self.matrix_factory = matrix_factory
        self.sampler = sampler
        self.first_min = []
        self.second_min = []
        self.third_min = []
        self.T = []
        self.logger = logging.getLogger("diploma")
        
    def compute_probabilities(self, time_bounds : tuple, **params):
        """
        Computes probabilities of ground and first two energy states.\n
        Arguments:\n
        time_bounds -- bounds for annealing time\n
        Params:\n
        k -- qubits per coordinate. If not set is counted automaticly\n
        num_reads -- number of sweaps on sampler per call
        """
        self.logger.info("Computing probabilities of ground and first two energy states\n" + 
            f"time_bounds={time_bounds}" + 
            f" num_reads={None if 'num_reads' not in params else params['num_reads']} " + 
            f"k={None if 'k' not in params else params['k']}")

        try:
            self.logger.info(f"Annealer type: {self.sampler.properties['chip_id']}")
        except:
            self.logger.warning("Unknown annealer type")

        self.logger.info(f"Input lattice: {self.B}\nInput target vector: {self.t}")

        self.first_min.clear()
        self.second_min.clear()
        self.third_min.clear()
        self.T.clear()

        for i in range(time_bounds[0], time_bounds[1]+1):
            k = None
            if "k" in params:
                k = params["k"]
                params.pop("k")
            solver = Solver(self.B, self.t, self.matrix_factory, self.sampler, k=k, gamma=1)
            params["annealing_time"] = 2**i
            res = solver.solve(**params)
            res.sort(key=lambda x: x.norm)
            self.T.append(2**i)
            self.first_min.append(res[0].prob)
            self.second_min.append(res[1].prob)
            if len(res) >= 3:
                self.third_min.append(res[2].prob)
            else:
                self.third_min.append(0)
            self.logger.debug(f"annealing time = {2**i} is done\n{list(map(lambda x: str(x), res))}")

        self.logger.debug(f"Plot info:\nfirst_min: {self.first_min}\n" +
            f"second_min: {self.second_min}\n" + 
            f"third_min: {self.third_min}\n" + 
            f"T: {self.T}")
    
    def show_plot(self, show=True, image_name=None, legend_loc="best", xlim=None, ylim=None):
        """
        Shows plot of perfomed computations\n
        Arguments:\n
        show - if true plot is shown (True by default)\n
        image_name - if not None saves plot in specified path (None by default)\n
        legend_loc - location of legend ('best' by default)\n
        xlim - tuple, limitation of x axis (None by default)\n
        ylim - tuple, limitation of y axis (None by default)\n
        """
        plt.clf()
        plt.xscale("log", base=2)
        
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
            
        plt.plot(self.T, self.first_min, ".b-", label=r"ground state")
        plt.plot(self.T, self.second_min, ".r-", label=r"first energy level")
        plt.plot(self.T, self.third_min, ".g-", label=r"second energy level")
        plt.legend(loc=legend_loc)
        
        if image_name is not None:
            plt.savefig(image_name, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

class StatCollectingInteractor(object):
    """
    Class computes average probabilities of ground and first two energy states
    for BDD problem with gamma=1 for sequence of matricies
    """

    def __init__(self, src, matrix_factory, sampler):
        """
        Arguments:\n
        src -- list of tuples of latticies and corresponding target vectors
        matrix_factory -- class for matrix generation\n
        sampler - annealing sampler
        """
        self.src = src
        self.matrix_factory = matrix_factory
        self.sampler = sampler
        self.T = None
        self.logger = logging.getLogger("diploma")

        self.first_min = None
        self.second_min = None
        self.third_min = None

    def compute_probabilities(self, time_bounds : tuple, **params):
        """
        Computes probabilities of ground and first two energy states.\n
        Arguments:\n
        time_bounds -- bounds for annealing time\n
        Params:\n
        k -- qubits per coordinate. If not set is counted automaticly\n
        num_reads -- number of sweaps on sampler per call
        """
        for B, t in self.src:
            interactor = TestInteractor(B, t, self.matrix_factory, self.sampler)
            interactor.compute_probabilities(time_bounds, **params)

            if self.T is None:
                self.T = array(interactor.T)
                self.first_min = array(interactor.first_min)
                self.second_min = array(interactor.second_min)
                self.third_min = array(interactor.third_min)
            else:
                self.first_min = vstack([self.first_min, interactor.first_min])
                self.second_min = vstack([self.second_min, interactor.second_min])
                self.third_min = vstack([self.third_min, interactor.third_min])
            
            self.logger.info(f"Computation for {B} and {t} is done")

        self.logger.debug(f"Plot info:\nfirst_min: {self.first_min}\n" +
            f"second_min: {self.second_min}\n" + 
            f"third_min: {self.third_min}\n" + 
            f"T: {self.T}")

    def show_plot(self, show=True, image_name=None, legend_loc="best", xlim=None, ylim=None):
        """
        Shows plot of perfomed computations\n
        Arguments:\n
        show - if true plot is shown (True by default)\n
        image_name - if not None saves plot in specified path (None by default)\n
        legend_loc - location of legend ('best' by default)\n
        xlim - tuple, limitation of x axis (None by default)\n
        ylim - tuple, limitation of y axis (None by default)\n
        """
        plt.clf()
        plt.xscale("log", base=2)
        
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
            
        plt.plot(self.T, average(self.first_min, axis=0), ".b-", label=r"ground state")
        plt.plot(self.T, average(self.second_min, axis=0), ".r-", label=r"first energy level")
        plt.plot(self.T, average(self.third_min, axis=0), ".g-", label=r"second energy level")
        plt.legend(loc=legend_loc)
        
        if image_name is not None:
            plt.savefig(image_name, dpi=300, bbox_inches='tight')
        if show:
            plt.show()