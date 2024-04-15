#!/usr/bin/env python
from ast import FunctionType
from multiprocessing import Pool
import multiprocessing
from types import MethodType
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

""" Pulled from scikit-opt sko.base module by @guofei9987"""

class SkoBase(metaclass=ABCMeta):
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
    PERIODIC = periodic
    REFLECTIVE = reflective
    RANDOM = random
    
def parallel_calc_init(locks_list):
    global locks_passed
    locks_passed = locks_list

def set_run_mode(func, mode):
    '''

    :param func:
    :param mode: string
        can be  common, vectorization , parallel, cached
    :return:
    '''
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
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''

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

class PSO_GA(SkoBase):
    def __init__(
        self,
        func,
        n_dim,
        config = None,
        F=0.5,
        size_pop=50,
        max_iter=200,
        lb=[-1000.0],
        ub=[1000.0],
        w=0.8,
        c1=0.1,
        c2=0.1,
        prob_mut=0.001,
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
        n_processes=1,
        taper_GA=False,
        taper_mutation=False,
        skew_social=True,
        early_stop=None,
        initial_guesses=None,
        guess_deviation=100,
        guess_ratio=0.25,
        vectorize_func=True,
        bounds_strategy:Bounds_Handler=Bounds_Handler.PERIODIC,
        mutation_strategy = 'DE/rand/1',
        func_args = None,
        verbose=False,
        pool_args = None,
        pass_particle_num=False,
        parallel_locks = None,
    ):
        self.func = func_transformer(func, n_processes, pass_worker_num=pass_particle_num) if config.get('vectorize_func', vectorize_func) else func  # , n_processes)
        self.func_raw = func
        self.n_processes = n_processes
        #print('func_args in init: '+str(func_args))
        self.func_args = func_args
        self.n_dim = n_dim
        self.locks = parallel_locks

        # if config_dict:
        self.F = config.get('F', F)
        assert config.get('size_pop', size_pop) % 2 == 0, "size_pop must be an even integer for GA"
        self.size_pop = config.get('size_pop', size_pop)
        self.tether_ratio = config.get('guess_ratio', guess_ratio)
        self.max_iter = config.get('max_iter', max_iter)
        self.prob_mut = config.get('prob_mut', prob_mut)
        self.early_stop = config.get('early_stop', early_stop)
        self.taper_GA = config.get('taper_GA', taper_GA)
        self.taper_mutation = config.get('taper_mutation', taper_mutation)
        self.skew_social = config.get('skew_social',  skew_social)
        self.bounds_handler:Bounds_Handler = config.get('bounds_strategy', bounds_strategy)
        self.mutation_strategy = config.get('mutation_strategy', mutation_strategy)
        self.pass_worker_num = pass_particle_num

        self.w = config.get('w', w)
        self.cp = config.get('c1', c1)  # personal best -- cognitive
        self.cg = config.get('c2', c2)  # global best -- social

        self.Chrom = None

        self.lb = np.array(config.get('lb',  lb))
        self.ub = np.array(config.get('ub',  ub))
        initial_guesses = config.get('initial_guesses', initial_guesses)
        guess_deviation = config.get('guess_deviation', guess_deviation)
        guess_ratio = config.get('guess_ratio', guess_ratio)

        # else:
        #     self.F = F
        #     assert size_pop % 2 == 0, "size_pop must be an even integer for GA"
        #     self.size_pop = size_pop
        #     self.tether_ratio = guess_ratio
        #     self.max_iter = max_iter
        #     self.prob_mut = prob_mut
        #     self.early_stop = early_stop
        #     self.taper_GA = taper_GA
        #     self.taper_mutation = taper_mutation
        #     self.skew_social = skew_social
        #     self.bounds_handler:Bounds_Handler = bounds_strategy
        #     self.mutation_strategy = mutation_strategy

        #     self.w = w
        #     self.cp = c1  # personal best -- cognitive
        #     self.cg = c2  # global best -- social

        #     self.Chrom = None

        #     self.lb, self.ub = np.array(lb), np.array(ub)

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

    def crt_X(self):
        tmp = np.random.rand(self.size_pop, self.n_dim)
        return tmp.argsort(axis=1)

    def crt_initial(
        self, initial_points=None, initial_deviation=1e2, tether_ratio=0.25
    ):
        # create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio
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
        self.X = self.X + self.V
        for particle, coord in enumerate(self.X):
            if (coord < self.lb).any() or (coord > self.ub).any():
                self.X[particle] = self.bounds_handler(self, coord, (self.lb, self.ub))


    def cal_y(self):

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
            self.Y = results.reshape(-1,1)#self.func(self.X).reshape(-1, 1)
        return self.Y

    def update_pbest(self):
        """
        personal best
        :return:
        """
        self.need_update = self.pbest_y > self.Y

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        """
        global best
        :return:
        """
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.X[idx_min, :].copy()
            self.gbest_y = float(self.pbest_y[idx_min])

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X)
        self.record_value["Y"].append(self.Y)

    def old_run(self, max_iter=None):
        self.max_iter = max_iter or self.max_iter
        for iter_num in range(self.max_iter):
            # self.update_V()
            self.recorder()
            self.update_X()
            # self.cal_y()
            # self.update_pbest()
            # self.update_gbest()

            if self.verbose:
                print(
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def de_iter(self):
        logger.log(logging.INFO, 'DE Iter starting...')
        self.mutation()
        self.recorder()
        self.crossover()
        self.selection()
        #self.cal_y() #TODO most recent: this should no longer be needed, selection will set correct Y values
        # old: ^ pass indices of selected particles and only recalculate those (needs new calc method)
        self.update_pbest()
        self.update_gbest()

    def pso_iter(self):
        logger.log(logging.INFO, 'PS Iter starting...')
        self.update_pso_V()
        self.recorder()
        old_x = self.X.copy()
        self.update_X()
        # if (old_x == self.X).all():
        #     print("this is unholy") #TODO make warning that X not changing
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def mutation(self):
        """
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        from differential evolution
        """
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
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
        """
        if rand < prob_crossover, use V, else use X
        """
        mask = np.random.rand(self.size_pop, self.n_dim) <= self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        """
        greedy selection
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

    def run(self, max_iter=None, precision=None, N=20):
        """
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        """
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso_iter()

            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
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

            if self.taper_mutation and iter_num == np.floor(0.25 * self.max_iter):
                self.prob_mut = self.prob_mut / 10.0
            elif self.taper_mutation and iter_num == np.floor(0.75 * self.max_iter):
                self.prob_mut = self.prob_mut / 10.0
            if self.skew_social and iter_num == np.floor(0.5 * self.max_iter):
                self.cg = self.cg + 0.25 * self.cp
                self.cp = self.cp * 0.75
            elif self.skew_social and iter_num == np.floor(0.75 * self.max_iter):
                self.cg = self.cg + (1/3) * self.cp
                self.cp = self.cp * (2/3)

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def chrom2x(self, Chrom):
        pass

    def ranking(self):
        pass

