from __future__ import absolute_import
from __future__ import division

import copy
import collections
import csv
import glob
import logging
import logging.config
import numpy as np
import os
import re
import sys

import calculate
import compare
import constants as co
import datatypes
import opt as opt
import parameters
import schrod_indep_filetypes
from hybrid_optimizer import PSO_GA

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)



class Swarm_Optimizer(opt.Optimizer):
    def __init__(
        self,
        direc=None,
        ff: schrod_indep_filetypes.FF = None,
        ff_lines=None,
        args_ff=None,
        args_ref=None,
        bias_to_current=True,
        loose_bounds = False
    ):
        super(Swarm_Optimizer, self).__init__(direc, ff, ff_lines, args_ff, args_ref)

        lower_bounds = []
        upper_bounds = []
        deviations = []
        for param in self.ff.params:
            match (param.ptype):
                case 'af':
                    lower_bounds.append(0.)
                    upper_bounds.append(10.)
                    deviations.append(0.25)
                case 'bf':
                    lower_bounds.append(-0.1)
                    upper_bounds.append(25.)
                    deviations.append(0.5)
                case 'ae':
                    lower_bounds.append(0.)
                    upper_bounds.append(180.)
                    deviations.append(np.inf)
                case 'be':
                    lower_bounds.append(0.)
                    upper_bounds.append(6.)
                    deviations.append(np.inf)
                case 'df':
                    lower_bounds.append(-5.)
                    upper_bounds.append(5.)
                    deviations.append(np.inf)
                case 'q':
                    lower_bounds.append(-6.)
                    upper_bounds.append(6.)
                    deviations.append(np.inf)
                case 'imp1'| 'imp2':
                    lower_bounds.append(0.)
                    upper_bounds.append(50.)
                    deviations.append(np.inf)
                case 'sb':
                    lower_bounds.append(0.)
                    upper_bounds.append(50.)
                    deviations.append(np.inf)
                case _ :
                    raise("Parameter type not supported: "+param.ptype)

        ff_params = [param.value for param in self.ff.params]

        self.opt_config = {
            "lb": lower_bounds,
            "ub": upper_bounds,
            "size_pop": 10,
            "vectorize_func": True,
            "taper_GA": True,
            "taper_mutation": True,
            "skew_social": True,
            "max_iter": 1000,
            "initial_guesses": ff_params if bias_to_current else None,
            "guess_deviation": deviations,
            "guess_ratio": 0.3,
            "mutation_strategy": "DE/rand/1",
        }
        if self.r_data is None:
            self.r_data = opt.return_ref_data(self.args_ref)

        self.r_dict = compare.data_by_type(self.r_data)

    def calculate_and_score(parameter_set, ref_dict) -> float:

        data = calculate.main(parameter_set)
        # deprecated
        # self.ff.score = compare.compare_data(r_data, data)
        c_dict = compare.data_by_type(data)
        r_dict = ref_dict
        r_dict, c_dict = compare.trim_data(r_dict, c_dict)
        score = compare.compare_data(r_dict, c_dict)
        return score

    # Don't worry that self.ff isn't included in self.new_ffs.
    # opt.catch_run_errors will know what to do if self.new_ffs
    # is None.
    @property
    def best_ff(self):
        return sorted(self.new_ffs, key=lambda x: x.score)[0]

    @opt.catch_run_errors
    def run(self, restart=None):
        """
        Once all attributes are setup as you so desire, run this method to
        optimize the parameters.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """
        self.hybrid_opt = PSO_GA(self.calculate_and_score, len(self.ff.params), config=self.opt_config, func_args=self.r_dict)

        # calculate initial FF results
        data = calculate.main(self.args_ff)
        c_dict = compare.data_by_type(data)
        r_dict, c_dict = compare.trim_data(self.r_dict, c_dict)
        self.ff.score = compare.compare_data(r_dict, c_dict)

        logger.log(20, "~~ HYBRID OPTIMIZATION ~~".rjust(79, "~"))
        logger.log(20, "INIT FF SCORE: {}".format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        self.best_ff_params, self.best_ff_score = self.hybrid_opt.run(precision=None, N=self._max_cycles_wo_change)

        #replace initial ff params with best params
        assert len(self.best_ff_params) == len(self.ff.params)
        self.best_ff: schrod_indep_filetypes.FF = copy.deepcopy(self.ff)
        self.best_ff.path = self.best_ff.path + ".hybrid.fld"
        for i, param in enumerate(self.best_ff.params):
            param.value = self.best_ff_params[i]

        logger.log(20, "BEST:")
        opt.pretty_ff_results(self.best_ff, level=20)
        logger.log(
            20, "~~ END HYBRID CYCLE ~~".rjust(79, "~")
        )

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

