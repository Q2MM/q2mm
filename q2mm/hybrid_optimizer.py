from abc import ABCMeta, abstractmethod
import numpy as np
from enum import Enum

from sko.PSO import PSO
from sko.DE import DE
from sko.base import SkoBase

from sko.tools import func_transformer
from sko.operators import crossover, mutation, ranking, selection
from sko.operators import mutation

# region Utilities

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
        n_processes=0,
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
        func_args = None
    ):
        self.func = func_transformer(func) if config.get('vectorize_func', vectorize_func) else func  # , n_processes)
        self.func_raw = func
        self.func_args = func_args
        self.n_dim = n_dim

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
            initial_points=initial_guesses,
            initial_deviation=guess_deviation,
            tether_ratio=guess_ratio,
        )
        v_high = self.ub - self.lb
        self.V = np.random.uniform(
            low=-v_high, high=v_high, size=(self.size_pop, self.n_dim)
        )
        self.Y = self.cal_y()
        self.pbest_x = self.X.copy()
        self.pbest_y = np.array([[np.inf]] * self.size_pop)

        self.gbest_x = self.pbest_x[0, :]
        self.gbest_y = np.inf
        self.gbest_y_hist = []
        self.update_gbest()
        self.update_pbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {"X": [], "V": [], "Y": []}
        self.verbose = False

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
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
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

    # def pso_add(self, c, x1, x2):
    #     x1, x2 = x1.tolist(), x2.tolist()
    #     ind1, ind2 = np.random.randint(0, self.n_dim - 1, 2)
    #     if ind1 >= ind2:
    #         ind1, ind2 = ind2, ind1 + 1

    #     part1 = x2[ind1:ind2]
    #     part2 = [i for i in x1 if i not in part1]  # this is very slow

    #     return np.array(part1 + part2)

    def update_pso_V(self):
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        self.V = (
            self.w * self.V
            + self.cp * r1 * (self.pbest_x - self.X)
            + self.cg * r2 * (self.gbest_x - self.X)
        )
        if (self.V == 0).all():
            print("uh oh")

    def update_X(self):
        self.X = self.X + self.V
        for particle, coord in enumerate(self.X):
            if (coord < self.lb).any() or (coord > self.ub).any():
                self.X[particle] = self.bounds_handler(self, coord, (self.lb, self.ub))



    # def tsp_update_X(self):
    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         x = self.pso_add(self.cp, x, self.pbest_x[i])
    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         x = self.pso_add(self.cg, x, self.gbest_x)
    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    #     for i in range(self.size_pop):
    #         x = self.X[i, :]
    #         new_x_strategy = np.random.randint(3)
    #         if new_x_strategy == 0:
    #             x = mutation.swap(x)
    #         elif new_x_strategy == 1:
    #             x = mutation.reverse(x)
    #         elif new_x_strategy == 2:
    #             x = mutation.transpose(x)

    #         self.X[i, :] = x

    #     self.cal_y()
    #     self.update_pbest()
    #     self.update_gbest()

    def cal_y(self):

        # calculate y for every x in X
        if self.func_args:
            self.Y = self.func(self.X, self.func_args).reshape(-1, 1)
        else:
            self.Y = self.func(self.X).reshape(-1, 1)
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
            self.gbest_y = self.pbest_y[idx_min]

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
        self.mutation()
        self.recorder()
        self.crossover()
        self.selection()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def pso_iter(self):
        self.update_pso_V()
        self.recorder()
        old_x = self.X.copy()
        self.update_X()
        if (old_x == self.X).all():
            print("this is unholy")
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

        match self.mutation_strategy:

            case 'DE/best/1':
                # DE/best/k strategy makes more sense here  (k=1 or 2)
                self.V = self.gbest_x + self.F * (X[r2, :] - X[r3, :])
            case 'DE/rand/1':
                self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])
            case 'DE/rand/2':
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
        f_X = (
            self.x2y().copy()
        )  # Uses x2y, which incorporates the constraint equations as a large penalty
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
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
                (
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


class HO(SkoBase, metaclass=ABCMeta):
    def __init__(
        self,
        func,
        n_dim=None,
        F=0.5,  # DE
        pop=40,  # pop in PSO, size_pop in GA
        max_iter=150,
        lb=-1e5,  # -1 in DE
        ub=1e5,  # 1 in DE
        initial_dev=1e2,
        tether_ratio=0.25,
        w=0.8,  # PSO
        c1=0.5,  # PSO
        c2=0.5,  # PSO
        constraint_eq=tuple(),
        constraint_ueq=tuple(),
        verbose=False,  # PSO
        dim=None,  # PSO
        prob_mut=0.3,  # DE
        early_stop=None,  # DE
        initial_points=None,
    ):
        super().__init__()
        self.pso: PSO = PSO(
            func,
            n_dim=n_dim,
            pop=pop,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            w=w,
            c1=c1,
            c2=c2,
            constraint_eq=constraint_eq,
            constraint_ueq=constraint_ueq,
            verbose=verbose,
            dim=dim,
        )
        self.de: DE = DE(
            func,
            n_dim=n_dim,
            F=F,
            size_pop=pop,
            max_iter=max_iter,
            lb=lb,
            ub=ub,
            prob_mut=prob_mut,
            constraint_eq=constraint_eq,
            constraint_ueq=constraint_ueq,
        )

        self.n_dim = n_dim or dim  # support the earlier version

        self.func = func_transformer(func)
        self.w = w  # inertia
        self.cp, self.cg = (
            c1,
            c2,
        )  # parameters to control personal best, global best respectively
        self.size_pop = pop  # number of particles
        self.n_dim = (
            n_dim  # dimension of particles, which is the number of variables of func
        )
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.F = F
        self.initial_points = initial_points
        self.initial_dev = initial_dev
        self.tether_ratio = tether_ratio

        self.lb, self.ub = np.array(lb) * np.ones(self.n_dim), np.array(ub) * np.ones(
            self.n_dim
        )
        assert (
            self.n_dim == len(self.lb) == len(self.ub)
        ), "dim == len(lb) == len(ub) is not True"
        assert np.all(self.ub > self.lb), "upper-bound must be greater than lower-bound"

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.is_feasible = np.array([True] * pop)

        # self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim))
        self.crtbp(
            initial_points=initial_points,
            initial_deviation=initial_dev,
            tether_ratio=tether_ratio,
        )

        v_high = self.ub - self.lb
        self.V = np.random.uniform(
            low=-v_high, high=v_high, size=(self.size_pop, self.n_dim)
        )  # speed of particles
        self.Y = self.cal_y()  # y = f(x) for all particles
        self.pbest_x = (
            self.X.copy()
        )  # personal best location of every particle in history
        self.pbest_y = np.array(
            [[np.inf]] * pop
        )  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(
            1, -1
        )  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()

        # record verbose values
        self.record_mode = False
        self.record_value = {"X": [], "V": [], "Y": []}
        self.best_x, self.best_y = (
            self.gbest_x,
            self.gbest_y,
        )  # history reasons, will be deprecated

        self.prob_mut = prob_mut  # probability of mutation
        self.early_stop = early_stop

        # constraint:
        # self.has_constraint = len(constraint_eq) > 0 or len(constraint_ueq) > 0
        # self.constraint_eq = list(constraint_eq)  # a list of equal functions with ceq[i] = 0
        # self.constraint_ueq = list(constraint_ueq)  # a list of unequal constraint functions with c[i] <= 0

        self.Chrom = None
        self.Y_raw = None  # shape = (size_pop,) , value is f(x)
        self.FitV = None  # shape = (size_pop,)

        # self.FitV_history = []
        self.generation_best_X = []
        self.generation_best_Y = []

        self.all_history_Y = []
        self.all_history_FitV = []

    # @abstractmethod
    # def chrom2x(self, Chrom):
    #     pass

    # @abstractmethod
    # def ranking(self):
    #     pass

    def crtbp(self, initial_points=None, initial_deviation=1e2, tether_ratio=0.25):
        # create the population and set it for the first round of PSO-GA
        assert 1 >= tether_ratio
        num_tethered = np.floor(self.size_pop * tether_ratio)
        if initial_points is not None:
            x_free = np.random.uniform(
                low=self.lb,
                high=self.ub,
                size=(int(self.size_pop - num_tethered), self.n_dim),
            )
            lower_tether = initial_points - initial_deviation
            upper_tether = initial_points + initial_deviation
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

    def mutation(self):
        """
        V[i]=X[r1]+F(X[r2]-X[r3]),
        where i, r1, r2, r3 are randomly generated
        """
        X = self.X
        # i is not needed,
        # and TODO: r1, r2, r3 should not be equal
        random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

        r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]
        # assert r1 != r2 and r2 != r3

        # 这里F用固定值，为了防止早熟，可以换成自适应值
        self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

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
        mask = np.random.rand(self.size_pop, self.n_dim) < self.prob_mut
        self.U = np.where(mask, self.V, self.X)
        return self.U

    def selection(self):
        """
        greedy selection
        """
        X = self.X.copy()
        f_X = self.x2y().copy()
        self.X = U = self.U
        f_U = self.x2y()

        self.X = np.where((f_X < f_U).reshape(-1, 1), X, U)
        return self.X

    def x2y(self):
        self.Y_raw = self.func(self.X)
        if not self.has_constraint:
            self.Y = self.Y_raw
        else:
            # constraint
            penalty_eq = np.array(
                [np.sum(np.abs([c_i(x) for c_i in self.constraint_eq])) for x in self.X]
            )
            penalty_ueq = np.array(
                [
                    np.sum(np.abs([max(0, c_i(x)) for c_i in self.constraint_ueq]))
                    for x in self.X
                ]
            )
            self.Y = self.Y_raw + 1e5 * penalty_eq + 1e5 * penalty_ueq
        return self.Y

    def old_run(self, max_iter=None, precision=None, N=20):
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.pso.X, self.pso.Y, self.pso.V = self.X, self.Y, self.V
            self.pso.recorder()
            self.pso.update_pbest()
            self.pso.update_gbest()
            self.pso.run(max_iter=1, precision=precision, N=N)
            self.X, self.Y, self.V = self.pso.X, self.pso.Y, self.pso.V
            self.recorder()
            self.pso.need_update
            self.update_pbest()
            self.update_gbest()
            de_y = self.Y.flatten()
            self.de.X, self.de.Y, self.de.V = self.X, self.Y.flatten(), self.V
            self.de.generation_best_X.append(self.gbest_x)
            self.de.generation_best_Y.append(self.gbest_y)
            self.de.all_history_Y.append(self.de.Y)
            self.de.run(max_iter=1)
            self.X, self.Y, self.V = (
                self.de.X,
                np.reshape(self.de.Y, (self.size_pop, 1)),
                self.de.V,
            )
            self.recorder()
            self.need_update
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print(
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def update_pso_V(self):
        r1 = np.random.rand(self.size_pop, self.n_dim)
        r2 = np.random.rand(self.size_pop, self.n_dim)
        self.V = (
            self.w * self.V
            + self.cp * r1 * (self.pbest_x - self.X)
            + self.cg * r2 * (self.gbest_x - self.X)
        )

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)  # TODO change boundary handler

    def cal_y(self):
        # calculate y for every x in X
        self.Y = self.func(self.X).reshape(-1, 1)
        return self.Y

    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_pbest(self):
        """
        personal best
        :return:
        """
        self.need_update = self.pbest_y > self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

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
            self.gbest_y = self.pbest_y[idx_min]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value["X"].append(self.X)
        self.record_value["V"].append(self.V)
        self.record_value["Y"].append(self.Y)

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
            if self.verbose:
                (
                    "Iter: {}, Best fit: {} at {}".format(
                        iter_num, self.gbest_y, self.gbest_x
                    )
                )
            self.gbest_y_hist.append(self.gbest_y)

            self.de_iter()

        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

    def de_iter(self):
        self.mutation()
        self.recorder()
        self.crossover()
        self.selection()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    def pso_iter(self):
        self.update_pso_V()
        self.recorder()
        self.update_X()
        self.cal_y()
        self.update_pbest()
        self.update_gbest()

    fit = run

