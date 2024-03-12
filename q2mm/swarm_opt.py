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
from hybrid_optimizer import PSO_GA, Bounds_Handler

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
        loose_bounds = False,
        ref_data=None,
        tight_spread=True
    ):
        super(Swarm_Optimizer, self).__init__(direc, ff, ff_lines, args_ff, args_ref)

        lower_bounds = []
        upper_bounds = []
        deviations = []
        for param in self.ff.params:
            if param.ptype == 'af':
                lower_bounds.append(0.1)
                upper_bounds.append(7.)
                deviations.append(0.125) if tight_spread else deviations.append(1.)
            elif param.ptype == 'bf':
                lower_bounds.append(0.1)
                upper_bounds.append(7.)
                deviations.append(0.125) if tight_spread else deviations.append(1.)
            elif param.ptype == 'ae':
                lower_bounds.append(0.)
                upper_bounds.append(180.)
                deviations.append(15.)
            elif param.ptype == 'be':
                lower_bounds.append(0.)
                upper_bounds.append(6.)
                deviations.append(.5) #TODO reassess
            elif param.ptype == 'df':
                lower_bounds.append(-5.)
                upper_bounds.append(5.)
                deviations.append(np.inf)
            elif param.ptype == 'q': #TODO MF - this may be removed bc charges will now always be parameterized with mjESP or mgESP in a linear fashion following P-ON 2024 work
                lower_bounds.append(-10.)
                upper_bounds.append(10.)
                deviations.append(2.) if param.value != 0 else deviations.append(10.)
            elif param.ptype == 'imp1'| param.ptype == 'imp2':
                lower_bounds.append(0.)
                upper_bounds.append(50.)
                deviations.append(np.inf)
            elif param.ptype == 'sb':
                lower_bounds.append(0.)
                upper_bounds.append(50.)
                deviations.append(np.inf)
            else:
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
            "max_iter": 10000,
            "initial_guesses": ff_params if bias_to_current else None,
            "guess_deviation": deviations,
            "guess_ratio": 0.7 if tight_spread else 0.3,
            "mutation_strategy": "DE/best/1",
        }
        #
        if ref_data is None:
            self.ref_data = opt.return_ref_data(self.args_ref)
        else:
            self.ref_data = ref_data
        self.r_dict = compare.data_by_type(self.ref_data)

        self.hybrid_opt = PSO_GA(self.calculate_and_score, len(self.ff.params), config=self.opt_config, func_args=self.r_dict, verbose=True, bounds_strategy=Bounds_Handler.REFLECTIVE)

    def calculate_and_score(self, parameter_set, ref_dict) -> float:

        #write out changes to fld file!!
        for param, new_val in zip(self.ff.params, parameter_set):
            param.value = new_val
        self.ff.export_ff()
        data = calculate.main(self.args_ff) #TODO this should be the og arguments to calculate, not the value of the parameters
        # deprecated
        # self.ff.score = compare.compare_data(r_data, data)
        c_dict = compare.data_by_type(data)
        r_dict = ref_dict
        r_dict, c_dict = compare.trim_data(r_dict, c_dict)
        score = compare.compare_data(r_dict, c_dict)
        self.ff.score = score
        logger.log(logging.INFO, "Score: "+str(score))
        return score

    # Don't worry that self.ff isn't included in self.new_ffs.
    # opt.catch_run_errors will know what to do if self.new_ffs
    # is None.
    @property
    def best_ff(self):
        return sorted(self.new_ffs, key=lambda x: x.score)[0]

    @opt.catch_run_errors
    def run(self, convergence_precision=None, ref_data=None, restart=None):
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

        self.best_ff_params, self.best_ff_score = self.hybrid_opt.run(precision=convergence_precision, max_iter=100)

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

