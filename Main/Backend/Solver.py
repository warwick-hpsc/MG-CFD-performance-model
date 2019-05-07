import pandas as pd
import numpy as np
import scipy.optimize as optimize
import statistics
import os, argparse
from pprint import pprint
from copy import deepcopy

class ModelFittingError(Exception):
  pass

def bound_step(**x):
    x_new = x["x_new"]
    return bool(np.all(x_new >= 0.0))
    # return True

class Solver(object):
	def __init__(self, conf, A, y, archModel):
		A_colnames = A.columns.values
		num_insn_coefs = len(A_colnames)
		num_coefs = num_insn_coefs
		if conf["do_spill_penalty"]:
		    num_coefs += 1
		if num_coefs > len(y):
		    raise ModelFittingError("Number of model coefficients ({0}) exceeds number of data points ({1}).".format(num_coefs , len(y)))

		self.y = y
		self.A = A
		self.A_colnames = A.columns.values.tolist()
		self.num_insn_coefs = len(self.A_colnames)

		self.extra_coef_names = []
		if conf["do_spill_penalty"]:
		    self.extra_coef_names.append("spill_penalty")

		self.am = archModel

		self.insn_ranges = {}
		self.insn_ranges["eu.alu"] = (1.0, None)
		self.insn_ranges["eu.simd_alu"] = (1.0, None)
		self.insn_ranges["eu.simd_shuffle"] = (1.0, None)
		self.insn_ranges["eu.fp_mov"] = (1.0, None)
		self.insn_ranges["eu.fp_add"] = (1.0, None)
		self.insn_ranges["eu.fp_mul"] = (1.0, None)
		self.insn_ranges["eu.fp_div_fast"] = (1.0, None)

		self.insn_ranges["mem.stores"] = (1.0, None)
		self.insn_ranges["mem.loads"] = (1.0, None)

		self.insn_ranges["eu.fp_div"] = (1.0, None)

	def gen_initial_guesses(self):
	    initial_guess_zero = []
	    initial_guess_sensible = []
	    initial_guess_sensible2 = []
	    initial_guess_sensible3 = []
	    for i in range(len(self.A_colnames)):
	        insn_name = self.A_colnames[i]
	        if insn_name == "eu.fp_div":
	        	g = 4.0
	        else:
	        	g = self.insn_ranges[insn_name][0]

	        initial_guess_sensible.append(g)
	        initial_guess_sensible2.append(g*2)
	        initial_guess_sensible3.append(g*3)
	        initial_guess_zero.append(0.0)

	    for coef in self.extra_coef_names:
	        initial_guess_sensible.append(0.5)
	        initial_guess_sensible2.append(0.5)
	        initial_guess_sensible3.append(0.5)
	        initial_guess_zero.append(0.0)

	    initial_guesses = [initial_guess_sensible]
	    # initial_guesses = [initial_guess_sensible, initial_guess_sensible2, initial_guess_sensible3]

	    return initial_guesses

	def calc_model_error_sum(self, x):
	    y_model = self.am.apply_model(x)
	    y_error = y_model - self.y.values
	    return np.dot(y_error, y_error)

	def check_coefs(self, coef_vals, initial_guess):
	    if (set(coef_vals).intersection(set(initial_guess)) == set(initial_guess)) \
	        and (set(coef_vals).intersection(set(initial_guess)) == set(coef_vals)):
	        ## Then 'coef_vals' == 'initial_guess'
	        # print(" Coefficients equal initial_guess")
	        return False

	    num_zeroes = 0
	    for c in coef_vals:
	        if c == 0.0 or c == -0.0:
	            num_zeroes += 1
	    if num_zeroes == len(coef_vals):
	        return False
	    else:
	        return True

	def find_solutions(self):
	    coef_bounds = []
	    for x in self.A_colnames:
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
	            coef_bounds.append((0, 16.0))
	        else:
	            coef_bounds.append((0, None))
	    coef_bounds = tuple(coef_bounds)

	    initial_guesses = self.gen_initial_guesses()

	    self.solutions = []

	    # my_stepsize=0.1
	    # my_stepsize=0.25
	    my_stepsize=0.5
	    # my_stepsize=1.0
	    # my_stepsize=2.0

	    A_colnames = self.A_colnames
	    # my_niters=10
	    # my_niters=20
	    # my_niters=50
	    # my_niters=100
	    # my_niters=150
	    # my_niters=200
	    # my_niters=250
	    # my_niters=len(A_colnames)*5
	    # my_niters=len(A_colnames)*10
	    # my_niters=len(A_colnames)*15
	    # my_niters=len(A_colnames)*20
	    # my_niters=int(round(pow(1.8, len(A_colnames))))
	    my_niters=40+int(round(pow(1.7, len(A_colnames))))

	    my_niters = int(round(my_niters / my_stepsize))

	    print("my_niters = {0}".format(my_niters))

	    # n_solver_iters = 10
	    # n_solver_iters = 25
	    # n_solver_iters = 50
	    # n_solver_iters = 75
	    n_solver_iters = 100

	    solver_options = {}
	    solver_options["L-BFGS-B"] = {}
	    solver_options["L-BFGS-B"]["maxiter"] = n_solver_iters
	    solver_options["L-BFGS-B"]["maxfun"] = n_solver_iters
	    # min_args={"method":"L-BFGS-B", "bounds":coef_bounds, "options":solver_options["L-BFGS-B"]}

	    solver_options["Nelder-Mead"] = {}
	    solver_options["Nelder-Mead"]["maxiter"] = n_solver_iters
	    solver_options["Nelder-Mead"]["maxfev"] = n_solver_iters

	    method="L-BFGS-B"
	    # method="Nelder-Mead"

	    min_args={"method":method, "options":solver_options[method]}
	    if method != "Nelder-Mead":
	        min_args["bounds"] = coef_bounds

	    np.random.seed(65432)
	    for ig in initial_guesses:
	        result = optimize.basinhopping(self.calc_model_error_sum, ig, niter=my_niters, stepsize=my_stepsize, 
	                                       accept_test=bound_step, 
	                                       minimizer_kwargs=min_args)
	        # if not result.lowest_optimization_result.success:
	        if not self.check_coefs(result.x, ig):
	            # print("{0} solver with initial guess {1} failed to find solution".format("basinhopping" ,ig.__str__()))
	            # print(" failed to find solution: " + result.message)
	            # sys.exit(-1)
	            pass
	        else:
	            r = result.lowest_optimization_result
	            # print("result:")
	            # pprint(r)
	            # quit()
	            self.solutions.append(r.x)

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
	    # (y_predict, do_contribute) = self.am.apply_model(solution, verify=True)
	    (y_predict, do_contribute) = self.am.apply_model(solution, do_print=True, verify=True)
	    for i in range(len(do_contribute)):
	    	if not do_contribute[i]:
	    		insn_name = self.A_colnames[i]
	    		cpi_estimate = solution[i]
	    		if cpi_estimate != self.insn_ranges[insn_name][0]:
	    			# print("Insn class '{0}' does not contribute to performance, but model has increased its value to '{1}' from minimum bound. Resetting it to minimum.".format(insn_name, cpi_estimate))
	    			solution[i] = self.insn_ranges[insn_name][0]

	    # coef_names = self.A_colnames + self.extra_coef_names
	    # solution_dict = {}
	    # for i in range(len(coef_names)):
	    # 	solution_dict[coef_names[i]] = solution[i]
	    # return solution_dict

	    return solution

	def get_coef_names(self):
		return self.A_colnames + self.extra_coef_names
