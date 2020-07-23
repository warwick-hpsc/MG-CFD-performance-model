import pandas as pd
import numpy as np
import scipy.optimize as optimize
import statistics, math
import os, argparse
from pprint import pprint
from copy import deepcopy

class ModelFittingError(Exception):
  pass

class Solver(object):
	def __init__(self, conf, A, y, archModel):
		A_colnames = A.columns.values
		
		# Update: treat spills as special case, using same CPI as mem.loads:
		if "mem.spills" in A_colnames:
			# Move to the end of frame to enable special treatment:
			A = A[[c for c in A_colnames if c != "mem.spills"] + ["mem.spills"]]

		num_insn_coefs = len(A_colnames)
		if "mem.spills" in A_colnames:
            # Ignore 'mem.spills'
		    num_insn_coefs -= 1
		num_coefs = num_insn_coefs
		if conf["do_spill_penalty"]:
			num_coefs += 1
		if conf["do_load_penalty"]:
			num_coefs += 1
		if num_coefs > len(y):
			raise ModelFittingError("Number of model coefficients ({0}) exceeds number of data points ({1}).".format(num_coefs , len(y)))

		self.y = y
		self.A = A
		self.A_colnames = A_colnames.tolist()

		self.num_insn_coefs = num_insn_coefs

		self.coef_names = A_colnames.tolist()
		if "mem.spills" in self.coef_names:
			self.coef_names.remove("mem.spills") # Ignore 'mem.spills'

		self.extra_coef_names = []
		if conf["do_spill_penalty"]:
			self.extra_coef_names.append("spill_penalty")
		if conf["do_load_penalty"]:
			self.extra_coef_names.append("load_penalty")

		self.conf = conf

		## Prepare optimisation parameters:
		if "optimisation_search_algorithm" in conf.keys():
			self.opt_method = conf["optimisation_search_algorithm"]
		else:
			# self.opt_method="basin"
			self.opt_method="shgo"
		self.basin_local_method = "L-BFGS-B"
		# self.basin_local_method = "Nelder-Mead"
		if "basin_local_iters" in conf.keys():
			self.basin_local_iters = int(conf["basin_local_iters"])
		else:
			self.basin_local_iters = int(round(500 + 200*math.log(num_coefs)))
			# self.basin_local_iters = int(round(500 + 400*math.log(num_coefs)))
		if "basin_jumps" in conf.keys():
			self.num_basin_jumps = int(conf["basin_jumps"])
		else:
			self.num_basin_jumps = 50 + int(round(50*math.log(num_coefs)))
		if "basin_steps" in conf.keys():
			self.basin_stepsize = int(conf["basin_steps"])
		else:
			self.basin_stepsize = 4

		self.am = archModel

		self.insn_ranges = {}
		self.insn_ranges["eu.alu"] = (1.0, None)
		self.insn_ranges["eu.simd_alu"] = (1.0, None)
		self.insn_ranges["eu.fp_add"] = (1.0, None)
		self.insn_ranges["eu.fp_mul"] = (1.0, None)
		self.insn_ranges["eu.fp_fma"] = (1.0, None)
		self.insn_ranges["eu.fp_div_fast"] = (1.0, None)
		self.insn_ranges["eu.fp_div"] = (1.0, None)
		self.insn_ranges["eu.fp_shuffle"] = (1.0, None)
		self.insn_ranges["eu.avx512"] = (1.0, None)
		self.insn_ranges["eu.simd_fp_add"] = (1.0, None)
		self.insn_ranges["eu.simd_fp_mul"] = (1.0, None)
		self.insn_ranges["eu.simd_fp_div"] = (1.0, None)

		self.insn_ranges["mem.stores"] = (1.0, None)
		self.insn_ranges["mem.loads"] = (1.0, None)
		# self.insn_ranges["mem.spills"] = (1.0, None)

		self.initial_guess = {}
		self.initial_guess["eu.alu"] = 1.0
		self.initial_guess["eu.simd_alu"] = 1.0
		# self.initial_guess["eu.simd_shuffle"] = 1.5
		# self.initial_guess["eu.fp_mov"] = 1.0
		self.initial_guess["eu.fp_add"] = 1.0
		self.initial_guess["eu.fp_mul"] = 1.0
		self.initial_guess["eu.fp_fma"] = 2.0
		self.initial_guess["eu.fp_div_fast"] = 4.0
		self.initial_guess["eu.fp_div"] = 8.0
		self.initial_guess["eu.fp_shuffle"] = 2.0
		self.initial_guess["eu.avx512"] = 1.0
		self.initial_guess["eu.simd_fp_add"] = 1.0
		self.initial_guess["eu.simd_fp_mul"] = 1.0
		self.initial_guess["eu.simd_fp_div"] = 1.0

		self.initial_guess["mem.stores"] = 1.0
		self.initial_guess["mem.loads"] = 3.0

	def gen_initial_guesses(self):
		initial_guess = []
		for i in range(len(self.coef_names)):
			insn_name = self.A_colnames[i]
			g = self.initial_guess[insn_name]
			initial_guess.append(g)

		for coef in self.extra_coef_names:
			initial_guess.append(1.0)

		return [initial_guess]

	def calc_model_error_sum(self, x):
		y_model = self.am.apply_model(x)
		y_error = y_model - self.y.values

		if not "predict_perf_diff" in self.conf.keys() or not self.conf["predict_perf_diff"]:
			## 'y_model' currently represents the ideal clock consumption 
			## of each Flux variant. Actual clock consumption is likely 
			## higher, but the difference should be similar for each kernel, 
			## as difference will be largely due to unstructured grid 
			## accesses which are identical across variants.
			## So, optimisation goal is for each prediction to have same absolute error:
			idle_cycles = self.y.values[0] - y_model[0]
			idle_cycles = max(idle_cycles, 0.0)
			y_error -= idle_cycles

		y_error_sum = np.dot(y_error, y_error)

		## Add fuzzy penalty for any coefficients out-of-range:
		for i in range(len(self.coef_names)):
			i_name = self.coef_names[i]
			i_min = self.insn_ranges[i_name][0]
			if (not i_min is None) and x[i] < i_min:
				y_error_sum += (100*(i_min - x[i]))**5
			i_max = self.insn_ranges[i_name][1]
			if (not i_max is None) and x[i] > i_max:
				y_error_sum += (100*(x[i]-i_max))**5

		return y_error_sum

	def check_coefs(self, coef_vals, initial_guess):
		if (set(coef_vals).intersection(set(initial_guess)) == set(initial_guess)) \
			and (set(coef_vals).intersection(set(initial_guess)) == set(coef_vals)):
			## Then 'coef_vals' == 'initial_guess'
			return False

		num_zeroes = 0
		for c in coef_vals:
			if c == 0.0 or c == -0.0:
				num_zeroes += 1
		return num_zeroes != len(coef_vals)

	def find_solutions(self):
		coef_bounds = []
		for x in self.coef_names:
			min_cpi = 0.0
			if not x in self.insn_ranges.keys():
				raise ModelFittingError("Cannot determine bound for insn '{0}'".format(x))
			min_cpi = self.insn_ranges[x][0]
			max_cpi = self.insn_ranges[x][1]
			coef_bounds.append((min_cpi, max_cpi))
		for coef in self.extra_coef_names:
			if "pct" in coef:
				coef_bounds.append((0, 1.0))
			elif coef == "spill_penalty":
				coef_bounds.append((0.0, 16.0))
			elif coef == "load_penalty":
				coef_bounds.append((0.0, 16.0))
			else:
				coef_bounds.append((0, None))
		coef_bounds = tuple(coef_bounds)
		num_coefs = len(coef_bounds)

		initial_guesses = self.gen_initial_guesses()
		self.solutions = []

		if self.opt_method == "basin":
			## Basin-hopping parameters. I have tuned these to provide a high probability 
			## that optimizer will complete converging to a solution, meaning that 
			## additional jumps or iterations do not improve further. 
			basin_step = self.basin_stepsize
			num_basin_jumps = self.num_basin_jumps
			num_local_iters = self.basin_local_iters
			basin_interval = 50
			print("Using basin-hopping, with {0} jumps, {1} step, {2} interval, and {3} minimise iters".format(num_basin_jumps, basin_step, basin_interval, num_local_iters))
			
			solver_options = {}
			solver_options["L-BFGS-B"] = {}
			solver_options["L-BFGS-B"]["maxiter"] = num_local_iters
			solver_options["L-BFGS-B"]["maxfun"]  = num_local_iters

			solver_options["Nelder-Mead"] = {}
			solver_options["Nelder-Mead"]["maxiter"] = num_local_iters
			solver_options["Nelder-Mead"]["maxfev"]  = num_local_iters

			local_args={"method":self.basin_local_method, "options":solver_options[self.basin_local_method]}

		elif self.opt_method == "shgo":
			print("Using shgo")

		np.random.seed(65432)
		for ig in initial_guesses:
			if self.opt_method == "basin":
				result = optimize.basinhopping(self.calc_model_error_sum, ig, niter=num_basin_jumps, interval=basin_interval, stepsize=basin_step, minimizer_kwargs=local_args)
			elif self.opt_method == "shgo":
				result = optimize.shgo(self.calc_model_error_sum, coef_bounds)
			# print(result.x)
			if not self.check_coefs(result.x, ig):
				print("{0} solver with initial guess {1} failed to find solution".format("basinhopping" ,ig.__str__()))
				pass
			else:
				self.solutions.append(result.x)

		if len(self.solutions)==0:
			raise ModelFittingError("No solutions found")

	def select_best_solution(self):
		solution_error_sums = [self.calc_model_error_sum(c) for c in self.solutions]
		# solution_error_maximums = [calc_model_error_max(c) for c in self.solutions]
		solutions_sorted = [b for a,b in sorted(zip(solution_error_sums, self.solutions), key=lambda p:p[0])]
		# return solutions_sorted[0]

		solution = solutions_sorted[0]

		## Filter out coefficients that do not contribute to overall performance of 
		## any run in fitting data:
		(y_predict, do_contribute) = self.am.apply_model(solution, do_print=True, verify=True)
		for i in range(len(self.coef_names)):
			insn_name = self.coef_names[i]
			cpi_estimate = solution[i]
			if not do_contribute[insn_name]:
				if cpi_estimate != self.insn_ranges[insn_name][0]:
					print("Insn class '{0}' does not contribute to performance, but model has increased its value to '{1}' from minimum bound. This is unjustified. Resetting it to minimum.".format(insn_name, cpi_estimate))
					solution[i] = self.insn_ranges[insn_name][0]
			if cpi_estimate < 0.0:
				solution[i] = self.insn_ranges[insn_name][0]

		for i in range(len(solution)):
			if solution[i] < 0.0:
				solution[i] = 0.0

		return solution

	def get_coef_names(self):
		return self.coef_names + self.extra_coef_names
