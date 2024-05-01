#!/usr/bin/env python
from ast import FunctionType
from multiprocessing import Pool
import multiprocessing
from types import MethodType
from typing import Tuple
import warnings
import sys
import numpy as np
from enum import Enum
import logging
#import constants as co

from functools import lru_cache, partial

from abc import ABCMeta, abstractmethod

#logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger('swarm_opt')


# region Utilities


class SkoBase(metaclass=ABCMeta):

    """ Pulled from scikit-opt sko.base module by @guofei9987 . This will be replaced by inheritance from
    @mmfarrugia generalized hybrid_optimizer which will inherit from this class in scikit-opt package TODO"""

    def register(self, operator_name, operator, *args, **kwargs):
        '''
        regeister udf to the class
        :param operator_name: string
        :param operator: a function, operator itself
        :param args: arg of operator
        :param kwargs: kwargs of operator
        :return:
        '''

        def operator_wapper(*wrapper_args):
            return operator(*(wrapper_args + args), **kwargs)

        setattr(self, operator_name, MethodType(operator_wapper, self))
        return self

    def fit(self, *args, **kwargs):
        warnings.warn('.fit() will be deprecated in the future. use .run() instead.'
                      , DeprecationWarning)
        return self.run(*args, **kwargs)


class Problem(object):
    pass

def reflective(self, position, bounds, **kwargs):
    r"""Reflect the particle at the boundary

    This method reflects the particles that exceed the bounds at the
    respective boundary. This means that the amount that the component
    which is orthogonal to the exceeds the boundary is mirrored at the
    boundary. The reflection is repeated until the position of the particle
    is within the boundaries. The following algorithm describes the
    behaviour of this strategy:

    .. math::
        :nowrap:

        \begin{gather*}
            \text{while } x_{i, t, d} \not\in \left[lb_d,\,ub_d\right] \\
            \text{ do the following:}\\
            \\
            x_{i, t, d} =   \begin{cases}
                                2\cdot lb_d - x_{i, t, d} & \quad \text{if } x_{i,
                                t, d} < lb_d \\
                                2\cdot ub_d - x_{i, t, d} & \quad \text{if } x_{i,
                                t, d} > ub_d \\
                                x_{i, t, d} & \quad \text{otherwise}
                            \end{cases}
        \end{gather*}
    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(position, bounds)
    new_pos = position
    while lower_than_bound[0].size != 0 or greater_than_bound[0].size != 0:
        if lower_than_bound[0].size > 0:
            new_pos[lower_than_bound] = (
                2 * lb[lower_than_bound[0]] - new_pos[lower_than_bound]
            )
        if greater_than_bound[0].size > 0:
            new_pos[greater_than_bound] = (
                2 * ub[greater_than_bound] - new_pos[greater_than_bound]
            )
        lower_than_bound, greater_than_bound = out_of_bounds(new_pos, bounds)

    return new_pos

def periodic(self, position, bounds, **kwargs):
    r"""Sets the particles a periodic fashion

    This method resets the particles that exeed the bounds by using the
    modulo function to cut down the position. This creates a virtual,
    periodic plane which is tiled with the search space.
    The following equation describtes this strategy:

    .. math::
        :nowrap:

        \begin{gather*}
        x_{i, t, d} = \begin{cases}
                            ub_d - (lb_d - x_{i, t, d}) \mod s_d & \quad \text{if }x_{i, t, d} < lb_d \\
                            lb_d + (x_{i, t, d} - ub_d) \mod s_d & \quad \text{if }x_{i, t, d} > ub_d \\
                            x_{i, t, d} & \quad \text{otherwise}
                      \end{cases}\\
        \\
        \text{with}\\
        \\
        s_d = |ub_d - lb_d|
        \end{gather*}

    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(
        position, bounds
    )
    lower_than_bound = lower_than_bound[0]
    greater_than_bound = greater_than_bound[0]
    bound_d = np.tile(
        np.abs(np.array(ub) - np.array(lb)), (position.shape[0], 1)
    )
    bound_d = bound_d[0]
    ub = np.tile(ub, (position.shape[0], 1))[0]
    lb = np.tile(lb, (position.shape[0], 1))[0]
    new_pos = position
    if lower_than_bound.size != 0:# and lower_than_bound[1].size != 0:
        new_pos[lower_than_bound] = ub[lower_than_bound] - np.mod(
            (lb[lower_than_bound] - new_pos[lower_than_bound]),
            bound_d[lower_than_bound],
        )
    if greater_than_bound.size != 0:# and greater_than_bound[1].size != 0:
        new_pos[greater_than_bound] = lb[greater_than_bound] + np.mod(
            (new_pos[greater_than_bound] - ub[greater_than_bound]),
            bound_d[greater_than_bound],
        )
    return new_pos

def random(self, position, bounds, **kwargs):
    """Set position to random location

    This method resets particles that exeed the bounds to a random position
    inside the boundary conditions.
    """
    lb, ub = bounds
    lower_than_bound, greater_than_bound = out_of_bounds(
        position, bounds
    )
    # Set indices that are greater than bounds
    new_pos = position
    new_pos[greater_than_bound[0]] = np.array(
        [
            np.array([u - l for u, l in zip(ub, lb)])
            * np.random.random_sample((position.shape[1],))
            + lb
        ]
    )
    new_pos[lower_than_bound[0]] = np.array(
        [
            np.array([u - l for u, l in zip(ub, lb)])
            * np.random.random_sample((position.shape[1],))
            + lb
        ]
    )
    return new_pos


def out_of_bounds(position, bounds):
    """Helper method to find indices of out-of-bound positions

    This method finds the indices of the particles that are out-of-bound.
    """
    lb, ub = bounds
    greater_than_bound = np.nonzero(position > ub)
    lower_than_bound = np.nonzero(position < lb)
    return (lower_than_bound, greater_than_bound)

class Bounds_Handler(Enum):
    """Handler enum to select the strategy to rectify values which are outside of the bounds of the search space.

    These strategies and implementations are adapted from @ljvmiranda96 's pyswarms package on github.

    Note:
        Periodic: treats the search space as an n-dimensional box with periodic boundary conditions.
        Reflective: treats the bounds of the search space as reflective walls, particles are reflected.
        Random: re-spawns out-of-bounds particles at a random position within the bounds of the search space.
    """    
    PERIODIC = periodic
    REFLECTIVE = reflective
    RANDOM = random
    
# def exp_decay(start:float, end:float, iter_num:int, max_iter:int, d2=7):
#     return (start - end - d1) * np.exp(1 / (1 + d2 * iter_num / max_iter))

def exp_decay(start:float, end:float, iter_num:int, max_iter:int, d2=7) -> float:
    """Exponential Decay function

    Args:
        start (float): starting value for the parameter under exponential decay.
        end (float): ending value for the parameter under exponential decay.
        iter_num (int): Current iteration number.
        max_iter (int): Maximum number of iterations
        d2 (int, optional): Constant which modulates the steepness of the exponential decay function. Defaults to 7.

    Returns:
        float: value of exponential decay function for the current iteration
    """    
    return (start - end) * np.exp(-d2 * iter_num / max_iter) + end

def parallel_calc_init(locks_list):
    #TODO remove this if it will never be used
    global locks_passed
    locks_passed = locks_list

def set_run_mode(func, mode):
    """Set the mode in which the function should be run by assigning it as an attribute to the function.

    Args:
        func (_type_): The function to be run/
        mode (str): the mode in which the function should be run. Typically 'multiprocessing'.
    """    
    if mode == 'multiprocessing' and sys.platform == 'win32':
        warnings.warn('multiprocessing not support in windows, turning to multithreading')
        mode = 'multithreading'
    if mode == 'parallel':
        mode = 'multithreading'
        warnings.warn('use multithreading instead of parallel')
    func.__dict__['mode'] = mode
    print('mode: '+str(mode))
    return

def func_transformer(func, n_processes, pass_worker_num = False):
    """Returns vectorized function capable of running in parallel with n_processes workers.

    Args:
        func (_type_): Input function which should be linear
        n_processes (_type_): number of processes to run in parallel when evaluating the vectorized function
        pass_worker_num (bool, optional): If true, the index of the worker in the pool will be passed to the vectorized function. Defaults to False.

    Returns:
        _type_: Vectorized function
    """    

    if pass_worker_num:

        # to support the former version
        if getattr(func, 'is_vector', False):
            warnings.warn('''
            func.is_vector will be deprecated in the future, use set_run_mode(func, 'vectorization') instead
            ''')
            set_run_mode(func, 'vectorization')

        mode = getattr(func, 'mode', 'others')
        valid_mode = ('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached', 'others')
        assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)
        if mode == 'vectorization':
            return func
        elif mode == 'cached':
            @lru_cache(maxsize=None)
            def func_cached(x, func_args=None):
                return func(x, func_args)

            def func_warped(X):
                return np.array([func_cached(tuple(x), i) for i, x in enumerate(X)])

            return func_warped
        elif mode == 'multithreading':
            assert n_processes >= 0, 'n_processes should >= 0'
            from multiprocessing.dummy import Pool as ThreadPool
            if n_processes == 0:
                pool = ThreadPool()
            else:
                pool = ThreadPool(n_processes)

            def func_transformed(X, func_args=None):
                return np.array(pool.map(func, enumerate(X)))

            return func_transformed
        elif mode == 'multiprocessing':
            assert n_processes >= 0, 'n_processes should >= 0'
            from multiprocessing import Pool
            if n_processes == 0:
                pool = Pool()
            else:
                pool = Pool(n_processes)
            def func_transformed(X, func_args=None):
                return np.array(pool.map(func, enumerate(X)))

            return func_transformed

        else:  # common
            def func_transformed(X, func_args=None):
                return np.array([func(x, func_args, i) for i, x in enumerate(X)])
        
    else:
        # to support the former version
        if (func.__class__ is FunctionType) and (func.__code__.co_argcount > 1):
            warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

            def func_transformed(X, func_args=None):
                return np.array([func(*tuple(x), func_args) for x in X])

            return func_transformed

        # to support the former version
        if (func.__class__ is MethodType) and (func.__code__.co_argcount > 2):
            warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

            def func_transformed(X, func_args=None):
                return np.array([func(tuple(x), func_args) for x in X])

            return func_transformed

        # to support the former version
        if getattr(func, 'is_vector', False):
            warnings.warn('''
            func.is_vector will be deprecated in the future, use set_run_mode(func, 'vectorization') instead
            ''')
            set_run_mode(func, 'vectorization')

        mode = getattr(func, 'mode', 'others')
        valid_mode = ('common', 'multithreading', 'multiprocessing', 'vectorization', 'cached', 'others')
        assert mode in valid_mode, 'valid mode should be in ' + str(valid_mode)
        if mode == 'vectorization':
            return func
        elif mode == 'cached':
            @lru_cache(maxsize=None)
            def func_cached(x, func_args=None):
                return func(x, func_args)

            def func_warped(X):
                return np.array([func_cached(tuple(x)) for x in X])

            return func_warped
        elif mode == 'multithreading':
            assert n_processes >= 0, 'n_processes should >= 0'
            from multiprocessing.dummy import Pool as ThreadPool
            if n_processes == 0:
                pool = ThreadPool()
            else:
                pool = ThreadPool(n_processes)

            def func_transformed(X, func_args=None):
                return np.array(pool.map(func, X))

            return func_transformed
        elif mode == 'multiprocessing':
            assert n_processes >= 0, 'n_processes should >= 0'
            from multiprocessing import Pool
            if n_processes == 0:
                pool = Pool()
            else:
                pool = Pool(n_processes)
            def func_transformed(X, func_args=None):
                return np.array(pool.map(func, X))

            return func_transformed

        else:  # common
            def func_transformed(X, func_args=None):
                print('func_args in func_transformed '+str(func_args))
                return np.array([func(x, func_args) for x in X])

    return func_transformed

# endregion Utilities

class PSO_DE(SkoBase):
    def __init__(
        self,
        func,
        n_dim:int,
        config:dict = None,
        F:Tuple[float, float]=(0.5, 0.5),
        size_pop:int=50,
        max_iter:int=200,
        lb:np.ndarray=[-1000.0],
        ub:np.ndarray=[1000.0],
        w:Tuple[float, float]=(0.9, 0.4),
        c1:Tuple[float, float]=(2.5,0.5),
        c2:Tuple[float, float]=(0.5, 2.5),
        recomb_constant:Tuple[float, float]=(0.7, 0.7),
        constraint_eq:tuple=tuple(),
        constraint_ueq:tuple=tuple(),
        n_processes:int=1,
        taper_GA:bool=False,
        early_stop:int=None,
        initial_guesses:np.ndarray=None,
        guess_deviation:np.ndarray=[100.],
        guess_ratio:float=0.25,
        vectorize_func:bool=True,
        bounds_strategy:Bounds_Handler=Bounds_Handler.PERIODIC,
        mutation_strategy:str = 'DE/rand/1',
        func_args = None,
        verbose:bool=False,
        pass_particle_num:bool=False,
        parallel_locks = None,
    ):
        """Creates a hybrid Particle Swarm (PS)-Differential Evolution (DE) Optimizer object and initializes the swarm.

        Note:
            Hyperparameter values F, recomb_constant, w, c1, and c2 can be varied strategically over the course of the
            optimization so provide both a starting and ending value for these parameters. The strategy for their variation
            is set by an argument given in the self.run method.

        Args:
            func (_type_): The heuristic function which evaluates particle fitness or, in other words, calculates the Y value for each particle.
            n_dim (int): # of dimensions of the search space/the particle positions.
            config (dict, optional): Dictionary to configure some or all optimizer hyperparameters/settings. Defaults to None.. Defaults to None.
            F (Tuple[float, float], optional): (start, end) differential weight or mutation constant for the DE step. Increasing this value increases the magnitude of the vectors which move the particles. Defaults to (0.5, 0.5).. Defaults to (0.5, 0.5).
            size_pop (int, optional): # of particles in the swarm. Defaults to 50.
            max_iter (int, optional): Maximum # of iterations. Defaults to 200.
            lb (np.ndarray, optional): lower bounds of search space. Accepts arguments of length n_dim or 1. If length is 1, the bound is used for all dimensions. Defaults to [-1000.0].
            ub (np.ndarray, optional): upper bounds of search space. Accepts arguments of length n_dim or 1. If length is 1, the bound is used for all dimensions.. Defaults to [1000.0].
            w (Tuple[float, float], optional): (start, end) inertial weight of particles for the PS step. Increasing this value encourages the particles to explore the search space. Defaults to (0.9, 0.4).
            c1 (Tuple[float, float], optional): (start, end) cognitive parameter for PS step. Represents the velocity bias towards a particle's personal best position, encourages exploration. Defaults to (2.5,0.5).
            c2 (Tuple[float, float], optional): (start, end) social parameter for PS step. Represents the velocity bias towards the swarm's global best position, encourages exploitation. Defaults to (0.5, 2.5).
            recomb_constant (Tuple[float, float], optional): (start, end) recombination constant or (DE, NOT GA) mutation probability for the DE step. Defaults to (0.7, 0.7).
            constraint_eq (tuple, optional): Constraint equality. Defaults to tuple().
            constraint_ueq (tuple, optional): Constraint inequality. Defaults to tuple().
            n_processes (int, optional): # of function evaluations to run in parallel. Defaults to 1.
            taper_GA (bool, optional): If True, the optimizer will decrease the frequency of DE steps in the optimization until they reach 0 at the end of optimization, running only PS steps. Defaults to False.
            early_stop (int, optional): _description_. Defaults to None. TODO
            initial_guesses (np.ndarray, optional): Starting point for the optimization of shape (n_dim, 1). Defaults to None.
            guess_deviation (np.ndarray, optional): If initial_points given, limits how far the initial position of the tethered particles will deviate from the initial points. Accepts arguments of length n_dim or 1 of float elements. If length is 1, the bound is used for all dimensions. Defaults to 100.
            guess_ratio (float, optional): The ratio of particles which should start at positions 'tethered' to the initial_points if given. Defaults to 0.25.
            vectorize_func (bool, optional): If True, the func argument method/heuristic function will be vectorized to calculate each particle's position independently/in parallel. Defaults to True.
            bounds_strategy (Bounds_Handler, optional): the bounds handler whose strategy should handle out-of-bounds particles. Defaults to Bounds_Handler.PERIODIC.
            mutation_strategy (str, optional): The mutation strategy which should be used for the DE steps. Defaults to 'DE/rand/1'.
            func_args (TODO, optional): Any additional arguments which should be used to evaluate the heuristic function. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.
            pass_particle_num (bool, optional): TODO if this is kept. Defaults to False.
            parallel_locks (_type_, optional): TODO if this is kept. Defaults to None.
        """        
        self.func = func_transformer(func, n_processes, pass_worker_num=pass_particle_num) if config.get('vectorize_func', vectorize_func) else func  # , n_processes)
        self.func_raw = func
        self.n_processes = n_processes
        #print('func_args in init: '+str(func_args))
        self.func_args = func_args
        self.n_dim = n_dim
        self.locks = parallel_locks # TODO if this is no longer anticipated, remove it

        # if config_dict:
        self.F_0, self.F_t = config.get('differential_weight', F)
        self.F = self.F_0
        assert config.get('size_pop', size_pop) % 2 == 0, "size_pop must be an even integer for GA"
        self.size_pop = config.get('size_pop', size_pop)
        self.tether_ratio = config.get('guess_ratio', guess_ratio)
        self.max_iter = config.get('max_iter', max_iter)
        self.recomb_constant_0, self.recomb_constant_t = config.get('recombination_constant', recomb_constant) # recombination constant or crossover probability, controls # of mutants which progress into next gen, lower for stability (fewer)
        self.recomb_constant = self.recomb_constant_0
        self.early_stop = config.get('early_stop', early_stop)
        self.taper_GA = config.get('taper_GA', taper_GA)
        self.taper_mutation = self.F_t != self.F_0 and self.F_t != self.F
        self.bounds_handler:Bounds_Handler = config.get('bounds_strategy', bounds_strategy)
        self.mutation_strategy = config.get('mutation_strategy', mutation_strategy)
        self.pass_worker_num = pass_particle_num #TODO: unsure if this will never be used to parallelize without copying files, remove if not anticipated

        self.w_0, self.w_t = config.get('inertia', w)
        self.w =  self.w_0
        self.cp_0, self.cp_t = config.get('cognitive', c1)
        self.cp = self.cp_0
        self.cg_0, self.cg_t = config.get('social', c2)  # global best -- social acceleration constant
        self.cg = self.cg_0
        self.skew_social = self.cg_0 != self.cg_t and self.cg_t != self.cg
        logger.log(logging.INFO, 'cp: {} w: {} cg {}'.format(self.cp, self.w, self.cg))

        self.Chrom = None

        self.lb = np.array(config.get('lb',  lb))
        self.ub = np.array(config.get('ub',  ub))
        initial_guesses = config.get('initial_guesses', initial_guesses)
        guess_deviation = config.get('guess_deviation', guess_deviation)
        guess_ratio = config.get('guess_ratio', guess_ratio)

        assert (
            self.n_dim == self.lb.size == self.ub.size
        ), "dim == len(lb) == len(ub) is not True"
        assert np.all(self.ub > self.lb), "upper-bound must be greater than lower-bound"

        self.has_constraint = bool(constraint_ueq) or bool(constraint_eq)
        self.constraint_eq = constraint_eq
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * size_pop)

        self.crt_initial(
            initial_points=np.array(initial_guesses),
            initial_deviation=guess_deviation,
            tether_ratio=guess_ratio,
        )
        v_high = self.ub - self.lb
        self.V = np.random.uniform(
            low=-v_high, high=v_high, size=(self.size_pop, self.n_dim)
        )
        #self.Y = self.cal_y() TODO this is only commented out for q2mm, because the ff pool of objects is not yet created
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        #self.update_gbest() TODO same as above
        #self.update_pbest() TODO same as above

        # record verbose values
        self.record_mode = True
        self.record_value = {"X": [], "V": [], "Y": []}
        self.verbose = verbose

    def crt_initial(
        self, initial_points:np.ndarray=None, initial_deviation:float=1e2, tether_ratio:float=0.25
    ):    
        """Populates the swarm with particles and their initial positions for the optimization, then handles any
        out-of-bounds particles.

        Args:
            initial_points (np.ndarray, optional): (# particles, # dimensions) matrix of starting positions for the swarm. Defaults to None.
            initial_deviation (float, optional): If initial_points given, limits how far the initial position of the tethered particles will deviate from the initial points. Defaults to 1e2.
            tether_ratio (float, optional): The ratio of particles which should start at positions 'tethered' to the initial_points if given. Defaults to 0.25.
        """          
        # create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio, "Invalid argument: tether_ratio must be less than or equal to 1."
        num_tethered = np.floor(self.size_pop * tether_ratio)
        if initial_points is not None:
            x_free = np.random.uniform(
                low=self.lb,
                high=self.ub,
                size=(int(self.size_pop - num_tethered), self.n_dim),
            )
            lower_tether = initial_points - np.ones(shape=(int(num_tethered), self.n_dim))*(initial_deviation)
            lower_tether = np.where(lower_tether < self.lb, self.lb, lower_tether)
                
            upper_tether = initial_points + initial_deviation
            upper_tether = np.where(upper_tether > self.ub, self.ub, upper_tether)
            
            x_tethered = np.random.uniform(
                low=lower_tether,
                high=upper_tether,
                size=(int(num_tethered), self.n_dim),
            )
            self.X = np.vstack((x_free, x_tethered))
        else:
            self.X = np.random.uniform(
                low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
            )
        for particle, coord in enumerate(self.X):
            if (coord < self.lb).any() or (coord > self.ub).any():
                self.X[particle] = self.bounds_handler(self, coord, (self.lb, self.ub))

    def update_pso_V(self):
        """Calculates new velocities (self.V) for the particles in the swarm.
        """        
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        self.V = (
            self.w * self.V
            + self.cp * r1 * (self.pbest_x - self.X)
            + self.cg * r2 * (self.gbest_x - self.X)
        )
        if (self.V == 0).all():
            print("uh oh") #TODO turn into a warning that indicates all are 0 and velocities are not going right again & why last time

    def update_X(self):
        """Updates the position (self.X) of the swarm particles based on their velocities (self.V),
        then ensures that all particles are within the bounds of the search space.
        """        
        self.X = self.X + self.V
        for particle, coord in enumerate(self.X):
            if (coord < self.lb).any() or (coord > self.ub).any():
                self.X[particle] = self.bounds_handler(self, coord, (self.lb, self.ub))


    def cal_y(self):
        """Calculate self.Y for each particle in the swarm (self.X). Evaluates the heuristic function but not the penalty function.

        Returns:
            np.ndarray: resulting self.Y
        """        
        # calculate y for every x in X
        if self.func_args is not None:
            partial_func = partial(self.func_raw, self.func_args)
            enumerated = enumerate(self.X)
            with Pool(self.n_processes) as pool: #, initializer=parallel_calc_init, initargs=(self.locks,)) as pool:
                results = np.array(pool.map(partial_func, enumerate(self.X)))
            self.Y = results.reshape(-1,1)#self.func(self.X, self.func_args).reshape(-1, 1)
        else:
            partial_func = partial(self.func_raw)
            with Pool(self.n_processes) as pool:
                results = np.array(pool.map(partial_func, enumerate(self.X)))
            self.Y = results.reshape(-1,1)#sel_type_f.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        """Updates the personal best value for each particle in the swarm.
        """        
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        """Updates the global best value found by any particle in the swarm.
        """        
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = float(self.pbest_y[idx_min])

    def recorder(self):
        """Records the current position and heuristic value of each particle in the swarm.

        This history is held by lists of np.ndarray in the self.record_value dictionary and values
        are only recorded if self.record_mode is True.
        """        
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X)
        self.record_value["Y"].append(self.Y)

    def de_iter(self):
        """Performs one iteration of Differential Evolution.
        """        
        logger.log(logging.INFO, 'DE Iter starting...')
        self.mutation()
        self.crossover()
        self.selection()
        #self.cal_y() #TODO most recent: this should no longer be needed, selection will set correct Y values
        # old: ^ pass indices of selected particles and only recalculate those (needs new calc method)
        self.update_pbest()
        self.update_gbest()
        self.recorder()

    def pso_iter(self):
        """Performs one iteration of Particle Swarm.
        """        
        logger.log(logging.INFO, 'PS Iter starting...')
        self.update_pso_V()
        old_x = self.X.copy()
        self.update_X()
        # if (old_x == self.X).all():
        #     print("this is unholy") #TODO make warning that X not changing
        self.cal_y()
        self.update_pbest()
        self.update_gbest()
        self.recorder()

    def mutation(self):
        """Performs a recombination, also known as a DE-style mutation (different
        from traditional Genetic Algorithm mutation) to calculate potential new positions for
        each particle in the swarm.

        V=X[r1]+F(X[r2]-X[r3]),
        where r1, r2, r3 are randomly generated

        Step 1/3 of DE's analog of PS's self.update_X method

        Returns:
            np.ndarray: self.V mutated/recombined positions for the DE step.
        """        

        X = self.X
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        while (r1 == r2).all() or (r2 == r3).all() or (r1 == r3).all():
            random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))
            r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

        if self.mutation_strategy == "DE/best/1":
            # DE/best/k strategy makes more sense here  (k=1 or 2)
            self.V = self.gbest_x + self.F * (X[r2, :] - X[r3, :])
        elif self.mutation_strategy == "DE/rand/1":
            self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])
        elif self.mutation_strategy == "DE/rand/2":
            self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

        # 这里F用固定值，为了防止早熟，可以换成自适应值

        # DE/either-or could also work

        # DE/cur-to-best/1 !!

        # DE/cur-to-pbest

        # the lower & upper bound still works in mutation
        mask = np.random.uniform(
            low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
        )
        self.V = np.where(self.V < self.lb, mask, self.V)
        self.V = np.where(self.V > self.ub, mask, self.V)
        return self.V

    def crossover(self):
        """Evaluates the probability of mutation/recombination for each particle in the swarm.
        
        Note:
        Compares a random number to the recombination constant to continue with a portion
        of the mutations/recombinations. The frequency of this is modulated by recomb_constant

        Step 2/3 of DE's analog of PS's self.update_X method

        Returns:
            np.ndarray: self.U the proposed new positions of the swarm.
        """        
        mask = np.random.rand(self.size_pop, self.n_dim) <= self.recomb_constant
        # if rand < prob_crossover, use V, else use X
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        """Performs a greedy selection of the proposed new positions of the swarm.

        Replaces current particle positions self.X with proposed particle positions self.U 
        only for particles where f(self.U) < f(self.X) = self.Y.

        Step 3/3 of DE's analog of PS's self.update_X method

        Returns:
            np.ndarray: self.X new particle positions
        """        
        X = self.X.copy()
        # f_X = (
        #     self.x2y().copy()
        # )  # Uses x2y, which incorporates the constraint equations as a large penalty
        f_X = self.Y_penalized if self.has_constraint else self.Y # shouldn't need to recalculate Y because the X values have not changed from the last PS iter yet
        self.X = U = self.U
        f_U = self.x2y() #TODO this already calculates Y, if can just make a Q2MM version inheriting from main, could then
        # reduce the number of times this needs to be recalculated by making ff-particles with stale flags

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U) #TODO could also just return the indices of those that changes and only recalculate those
        self.Y = np.where((f_X < f_U).reshape(-1, 1), f_X, f_U) #TODO most recent: this should eliminate the need to recalculate Y after selection
        return self.X

    def x2y(self):
        """Evaluates the heuristic function (self.Y) AND any constraint functions (self.Y_penalized).

        Returns:
            np.ndarray: self.Y_penalized if constraints exist, else self.Y
        """        
        self.cal_y()
        if self.has_constraint:
            penalty_eq = 1e5 * np.array(
                [
                    np.array([np.sum(np.abs([c_i(x) for c_i in self.constraint_eq]))])
                    for x in self.X
                ]
            )
            penalty_eq = np.reshape(penalty_eq, (-1, 1))
            penalty_ueq = 1e5 * np.array(
                [
                    np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq]))
                    for x in self.X
                ]
            )
            penalty_ueq = np.reshape(penalty_ueq, (-1, 1))
            self.Y_penalized = self.Y + penalty_eq + penalty_ueq
            return self.Y_penalized
        else:
            return self.Y

    def run(self, max_iter:int=None, precision:float=None, N:int=20, strategy:str='exp_decay') -> Tuple[np.ndarray, float]:
        """Run the hybrid optimizer until maximum iterations or precision reached.

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to None and uses self.max_iter else uses max_iter and replaces sef.max_iter.
            precision (float, optional): If precision is None, it will run the number of max_iter steps. If precision is a float, the loop will stop if continuous N difference between pbest less than precision. Defaults to None.
            N (int, optional): # of stagnant iterations before precision is considered reached. Defaults to 20.
            strategy (str, optional): Strategy by which to vary optimization hyperparamaters. Defaults to 'exp_decay'.

        Returns:
            Tuple[np.ndarray, float]: (best position, best heuristic value) results of optimization
        """        

        self.max_iter = max_iter or self.max_iter
        logger.log(logging.INFO, 'max iter {}'.format(self.max_iter))
        c = 0
        print(str(strategy))
        for iter_num in range(self.max_iter):
            self.pso_iter()

            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    logger.log(logging.INFO, 'precision ({}) greater than tor_iter ({})'.format(precision, tor_iter))
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.taper_GA:
                if (
                    iter_num <= np.floor(0.25 * self.max_iter)
                    or (
                        iter_num <= np.floor(0.75 * self.max_iter)
                        and iter_num % 10 == 0
                    )
                    or (iter_num % 100 == 0)
                ):
                    self.de_iter()
            else:
                self.de_iter()

            if self.verbose:
                logger.log(logging.INFO,
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )
            self.gbest_y_hist.append(self.gbest_y)

            if self.taper_mutation:
                if strategy == 'exp_decay':
                    self.F = exp_decay(self.F_0, self.F_t, iter_num, self.max_iter)
                    logger.log(logging.INFO, 'F; {}'.format(self.F))
            if self.skew_social:
                if strategy == 'exp_decay':
                    self.cp = exp_decay(self.cp_0, self.cp_t, iter_num, self.max_iter)
                    self.w = exp_decay(self.w_0, self.w_t, iter_num, self.max_iter)
                    self.cg = (self.cg_0 + self.cp_0) - self.cp
            logger.log(logging.INFO, 'cp: {} w: {} cg {}'.format(self.cp, self.w, self.cg))

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass

