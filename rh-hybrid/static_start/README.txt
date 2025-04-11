All calculations here are run with mm3.fld as rh_hyd_enamide_q_seminario.fld as mm3.fld, so fuerza estimated force constants with the dipole moments determined from Patrick Donoghue's published work on the Rh Hydrogenation of Enamides TSFF. This is both to remove any misguiding effects from unparameterized charges and also to properly simulate where within the Q2MM workflow the hybrid optimizer is used: after parameterization of charges. 3 technical replicates were run for each case and are under numbered folders within their relevant case.

Hyperparameters:
	All are run with literature-derived hyperparameters tweaked for a tighter exploration of parameterization surface, thus the TIGHT_OPT_CONFIG dictionary of settings in the code.

Cases evaluated:
	PSO only
	DE only
	Normal HO with tapering of DE step frequency at the end of the cycle/run
	Normal HO with tapering of DE step frequency at end of first cycle then no further DE
	Normal HO with no tapering of DE step frequency at all
	
	After investigating below issue, if it seems to be a disruption of the particle momentum, try DE steps only in direction of momentum after end of first cycle?
		Normal HO with no tapering of DE step frequency at all but steps only in line with particle momentum

Need to still investigate potential bug:
	How are particle scores getting worse when DE is run?
		Is it because they disrupt the PSO momentum or because greedy selection is improperly implemented?

