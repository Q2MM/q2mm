from __future__ import absolute_import
from __future__ import division

import copy
import logging
import logging.config
import multiprocessing
import numpy as np
import os

import calculate
import compare
import constants as co
import datatypes
import opt as opt
from schrod_dep_stuff import check_licenses
import schrod_indep_filetypes
from hybrid_optimizer import PSO_DE, Bounds_Handler, set_run_mode

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# region SWARM OPTIMIZER HYPERPARAMETER CONFIGURATIONS
TIGHT_SEARCH_CONFIG = {
    "vectorize_func": False,
    "taper_GA": False,
    "taper_mutation": True,
    "skew_social": True,
    "mutation_strategy": "DE/best/1",
    "differential_weight": (0.4, 0.1),
    "recomb_constant": (0.7, 0.7),
    "inertia": (0.7, 0.4),  # LDIW strategy
    "cognitive": (2.0, 0.5),
    "social": (1.0, 2.5),
}

GLOBAL_SEARCH_CONFIG = (
    {  # NOTE: user should also increase population size, global has slower convergence
        "vectorize_func": False,
        "taper_GA": False,
        "taper_mutation": True,
        "skew_social": True,
        "mutation_strategy": "DE/best/1",
        "differential_weight": (0.7, 0.1),
        "recomb_constant": (0.7, 0.7),
        "inertia": (0.9, 0.4),  # LDIW strategy
        "cognitive": (2.5, 0.5),
        "social": (0.5, 2.5),
    }
)
# widens the search radius but slows convergence

# endregion


class Swarm_Optimizer(opt.Optimizer):
    def __init__(
        self,
        direc=None,
        ff: schrod_indep_filetypes.FF = None,
        ff_lines=None,
        args_ff=None,
        args_ref=None,
        bias_to_current=True,
        loose_bounds=False,
        ref_data=None,
        tight_spread=True,
        num_ho_cores=1,
        max_iter=1000,
        pop_size=24,
    ):
        super(Swarm_Optimizer, self).__init__(direc, ff, ff_lines, args_ff, args_ref)

        lower_bounds = []
        upper_bounds = []
        deviations = []
        for param in self.ff.params:
            if param.ptype == "af":
                lower_bounds.append(0.1)
                upper_bounds.append(7.0)
                deviations.append(0.125) if tight_spread else deviations.append(1.0)
            elif param.ptype == "bf":
                lower_bounds.append(0.1)
                upper_bounds.append(7.0)
                deviations.append(0.125) if tight_spread else deviations.append(1.0)
            elif param.ptype == "ae":
                lower_bounds.append(0.0)
                upper_bounds.append(180.0)
                deviations.append(15.0)
            elif param.ptype == "be":
                lower_bounds.append(0.0)
                upper_bounds.append(6.0)
                deviations.append(0.5)  # TODO reassess
            elif param.ptype == "df":
                lower_bounds.append(-5.0)
                upper_bounds.append(5.0)
                deviations.append(np.inf)
            elif (
                param.ptype == "q"
            ):  # TODO MF - this may be removed bc charges will now always be parameterized with mjESP or mgESP in a linear fashion following P-ON 2024 work
                lower_bounds.append(-10.0)
                upper_bounds.append(10.0)
                deviations.append(2.0) if param.value != 0 else deviations.append(10.0)
            elif param.ptype == "imp1" | param.ptype == "imp2":
                lower_bounds.append(0.0)
                upper_bounds.append(50.0)
                deviations.append(np.inf)
            elif param.ptype == "sb":
                lower_bounds.append(0.0)
                upper_bounds.append(50.0)
                deviations.append(np.inf)
            else:
                raise ("Parameter type not supported: " + param.ptype)

        ff_params = [param.value for param in self.ff.params]

        param_opt_config = {
            "lb": lower_bounds,
            "ub": upper_bounds,
            "size_pop": pop_size,
            "max_iter": max_iter,
            "initial_guesses": ff_params if bias_to_current else None,
            "guess_deviation": deviations,
            "guess_ratio": 0.7 if tight_spread else 0.3,
        }
        self.opt_config = (
            {**param_opt_config, **TIGHT_SEARCH_CONFIG}
            if tight_spread
            else {**param_opt_config, **GLOBAL_SEARCH_CONFIG}
        )

        if ref_data is None:
            self.ref_data = opt.return_ref_data(self.args_ref)
        else:
            self.ref_data = ref_data
        self.r_dict = compare.data_by_type(self.ref_data)

        if num_ho_cores >= 1:
            self.setup_schrod_licenses(num_ho_cores)
        # if num_ho_cores > 1:
        #    assert struct_file_locks is not None, "If processing in parallel, there must be protections (locks) on shared resources such as structure files."
        set_run_mode(self.calculate_and_score, "multiprocessing")

        self.hybrid_opt = PSO_DE(
            self.calculate_and_score,
            len(self.ff.params),
            config=self.opt_config,
            func_args=self.r_dict,
            n_processes=self.num_ff_threads,
            pass_particle_num=True,
            verbose=True,
            bounds_strategy=Bounds_Handler.REFLECTIVE,
        )

        self.setup_ff_pool()

        self.hybrid_opt.Y = self.hybrid_opt.cal_y()
        self.hybrid_opt.update_pbest()
        self.hybrid_opt.update_gbest()
        self.hybrid_opt.recorder()

    def setup_schrod_licenses(self, num_cores: int):
        macro_avail, suite_avail = check_licenses()
        div_suite = suite_avail  # / co.MIN_SUITE_TOKENS
        div_macro = macro_avail  # / co.MIN_MACRO_TOKENS
        num_ff_threads = int(np.floor(min([div_suite, div_macro])))
        assert (
            num_cores <= multiprocessing.cpu_count()
        )  # TODO MF - add descriptive exception message
        if num_cores > self.opt_config.get("size_pop"):
            num_cores = self.opt_config.get("size_pop")
        self.num_ff_threads = min(num_cores, num_ff_threads)

        self.base_pool_dir = self.direc
        for i in range(self.num_ff_threads):
            os.mkdir(os.path.join(self.direc, "temp_" + str(i)))
            os.popen(
                "cp "
                + os.path.join(self.direc, "*.mae")
                + " "
                + os.path.join(self.direc, "temp_" + str(i))
            )
            os.popen(
                "cp "
                + os.path.join(self.direc, "*.out")
                + " "
                + os.path.join(self.direc, "temp_" + str(i))
            )
            os.popen(
                "cp "
                + os.path.join(self.direc, "atom.typ")
                + " "
                + os.path.join(self.direc, "temp_" + str(i))
            )

        # pool_args = {'num_workers': num_ff_threads, 'base_path':base_pool_dir}

    def setup_ff_pool(self):
        self.pool_ff_objects = list()
        for i in range(self.opt_config.get("size_pop")):
            j = i % self.num_ff_threads
            ff_i: datatypes.FF = copy.deepcopy(self.ff)
            ff_i.set_param_values(self.hybrid_opt.X[i])
            ff_i.path = os.path.join(self.base_pool_dir, "temp_" + str(j), "mm3.fld")
            self.pool_ff_objects.append(ff_i)

            # then the export of the ff will take care of itself, will just need to pass worker num to calculate and score

    def calculate_and_score(self, ref_dict, enumerable_input) -> float:
        # TODO MF the HO should really primarily be used with parallel processing, but I SHOULD make a separate function that does this for parallel, just set func for PSO_DE different based on n_processors flag
        # for now assume multiprocessing but perhaps later include the windows supported method of multithreading but might not need bc partial

        ff_num, parameter_set = enumerable_input
        if ff_num is not None:
            ff = self.pool_ff_objects[ff_num]
            logger.log(logging.DEBUG, "FF Num: " + str(ff_num))
        else:
            ff = self.ff

        # write out changes to fld file!!
        ff.set_param_values(parameter_set)
        if ff.stale_file:
            ff.export_ff()
        if ff.stale_score:
            ff_direc = os.path.dirname(ff.path)
            os.chdir(ff_direc)
            calc_args = copy.deepcopy(self.args_ff)
            calc_args[1] += "/temp_" + str(ff_num % self.num_ff_threads)
            # calc_args.append(['-d', './temp_'+str(ff_num)])
            data = calculate.main(calc_args)
            os.chdir(self.direc)
            c_dict = compare.data_by_type(data)
            r_dict = ref_dict
            r_dict, c_dict = compare.trim_data(r_dict, c_dict)
            score = compare.compare_data(r_dict, c_dict)
            ff.set_new_score(float(score))

        logger.log(logging.INFO, "FF " + str(ff_num) + " Score: " + str(score))
        print("FF " + str(ff_num) + " Score: " + str(score))
        return ff.get_score()

    @opt.catch_run_errors
    def run(
        self,
        convergence_precision=None,
        ref_data=None,
        restart=None,
        strategy="exp_decay",
    ):
        """
        Once all attributes are setup as you so desire, run this method to
        optimize the parameters.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """

        # calculate initial FF results
        data = calculate.main(self.args_ff)
        c_dict = compare.data_by_type(data)
        r_dict, c_dict = compare.trim_data(self.r_dict, c_dict)
        self.ff.score = compare.compare_data(r_dict, c_dict)

        logger.log(20, "~~ HYBRID OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        self.best_ff_params, self.best_ff_score = self.hybrid_opt.run(
            precision=convergence_precision, strategy=strategy
        )

        # replace initial ff params with best params
        assert len(self.best_ff_params) == len(self.ff.params)

        self.best_ff = copy.deepcopy(
            self.ff
        )  # schrod_indep_filetypes.FF(path=(self.ff.path[:-4] +'.hybrid.fld'), params=self.best_ff_params, score=self.best_ff_score)
        self.best_ff.path = self.best_ff.path[:-4] + ".hybrid.fld"
        self.best_ff.set_param_values(self.best_ff_params)
        self.best_ff.score = self.best_ff_score

        logger.log(20, "BEST:")
        opt.pretty_ff_results(self.best_ff, level=20)
        logger.log(20, "~~ END HYBRID CYCLE ~~".rjust(79, "~"))

        if self.best_ff.score < self.ff.score:
            logger.log(20, "~~ HYBRID FINISHED WITH IMPROVEMENTS ~~".rjust(79, "~"))
        else:
            logger.log(20, "~~ HYBRID FINISHED WITHOUT IMPROVEMENTS ~~".rjust(79, "~"))
            # This restores the inital parameters, so no need to use
            # restore_simp_ff here.
            self.best_ff = self.ff

        opt.pretty_ff_results(self.ff, level=20)
        opt.pretty_ff_results(self.best_ff, level=20)
        logger.log(20, "  -- Writing best force field from Hybrid Optimization.")
        self.best_ff.export_ff(self.best_ff.path)
        return self.best_ff
