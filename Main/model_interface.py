import pandas as pd
import numpy as np
import scipy.optimize as optimize
import statistics
import os, sys, argparse, csv
from pprint import pprint
from copy import deepcopy

import imp
imp.load_source('ArchModel', os.path.join(os.path.dirname(os.path.realpath(__file__)), "Backend", "ArchModel.py"))
from ArchModel import *
imp.load_source('Solver', os.path.join(os.path.dirname(os.path.realpath(__file__)), "Backend", "Solver.py"))
from Solver import *
imp.load_source('Utils', os.path.join(os.path.dirname(os.path.realpath(__file__)), "Utils.py"))
from Utils import *

script_dirpath = os.path.dirname(os.path.realpath(__file__))

y_name = "wg_cycles"
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='fit', action='store_true')
    parser.add_argument('-p', help='predict', action='store_true')
    args = parser.parse_args()

    init_np()

    conf = init_conf()

    if args.f:
        y, A = load_fitting_data()
        am = ArchModel(conf, A)
        solution = find_solution(conf, A, y, am)
        write_solution(solution)

    elif args.p:
        coefs = load_coefficients()
        y, A = load_predict_data()
        am = ArchModel(conf, A)
        y_predict = am.predict(coefs)
        y_predict = y_predict[0]

        wg_cycles_reference = load_validation_data()
        if wg_cycles_reference != None:
            y_predict = max(y_predict, wg_cycles_reference["rw_cycles"])
        
        write_prediction(conf, y_predict)

def init_np():
    float_formatter = lambda x: "%+.2E" % x
    # float_formatter = lambda x: "%+.1E" % x
    # float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter}, linewidth=np.nan)

    # np.random.seed(123456)
    np.random.seed(65432)

def init_conf():
    conf_filepath = os.path.join("Modelling", "insn_model_conf.csv")

    ## Model config:
    conf = {}
    conf["do_spill_penalty"] = False
    conf["cpu_is_skylake"] = False
    conf["cpu_is_broadwell"] = False
    conf["cpu_is_haswell"] = False
    conf["cpu_is_westmere"] = False
    conf["cpu_is_knl"] = False
    conf["avx512_simd_enabled"] = False
    conf["predict_perf_diff"] = False
    conf["predict_perf_direction_additive"] = False

    conf_df = pd.read_csv(conf_filepath, header=0)
    for idx, row in conf_df.iterrows():
        k = row["key"]
        v = row["value"]
        if v == "TRUE":
            conf[k] = True
        elif v == "FALSE":
            conf[k] = False
        else:
            conf[k] = v

    return conf

def load_data(filepath):
    data = pd.read_csv(filepath, header=0)

    eu_names = [n for n in data.columns.values if "eu" in n]
    mem_names = [n for n in data.columns.values if "mem" in n]
    coef_names = eu_names + mem_names

    ## Remove duplicate rows:
    data = data.drop_duplicates(coef_names)

    y = data[y_name]
    A = data[coef_names]

    return y, A

def load_fitting_data():
    fitting_data_filepath = os.path.join("Modelling", "fitting_data.csv")
    return load_data(fitting_data_filepath)

def load_predict_data():
    filepath = os.path.join("Modelling", "prediction_data.csv")
    return load_data(filepath)

def load_validation_data():
    filepath = os.path.join("Modelling", "validate_prediction_data.csv")
    if os.path.isfile(filepath):
        data = pd.read_csv(filepath, header=0)
        
        return {k:data[k][0] for k in ["mini_cycles", "correct_cycles", "rw_cycles"]}
    else:
        return None

def find_solution(conf, A, y, am):
    s = Solver(conf, A, y, am)
    s.find_solutions()
    coef_final_estimate = s.select_best_solution()

    ## Measure model error against fitting data:
    y_model = am.apply_model(coef_final_estimate, get_meta_coefs(conf, coef_final_estimate))
    y_error = [ b-a for a,b in zip(y, y_model)]
    # print("y_error = {0}".format(y_error))
    y_error_pct = [ float(e)/a for e,a in zip(y_error, y)]
    print("y_error_pct:")
    print(["{0}%".format(round(100.0*p, 1)) for p in y_error_pct])
    # y_error_sum = s.calc_model_error_sum(coef_final_estimate)
    # print("y_error_sum = {0}".format(y_error_sum))

    # return coef_final_estimate

    coef_names = s.get_coef_names()
    solution_dict = {}
    for i in range(len(coef_names)):
        solution_dict[coef_names[i]] = coef_final_estimate[i]
    return solution_dict


def write_solution(coef_final_estimate):
    solution_filepath = os.path.join("Modelling", "solution.csv")
    if os.path.exists(solution_filepath):
        os.remove(solution_filepath)

    with open(solution_filepath, "w") as solution_file:
        solution_file.write("coef,cpi\n")
        for k in coef_final_estimate:
            # cpi = coef_final_estimate[k]
            cpi = round(coef_final_estimate[k], 2)
            # cpi = round(coef_final_estimate[k], 3)
            solution_file.write("{0},{1}\n".format(k, cpi))

def write_prediction(conf, cycles_prediction):
    filepath = os.path.join("Modelling", "prediction.csv")
    if os.path.exists(filepath):
        os.remove(filepath)

    header="cycles_model"
    data_line="{0}".format(cycles_prediction)

    wg_cycles_reference = load_validation_data()
    if wg_cycles_reference != None:
        cycles_correct = wg_cycles_reference["correct_cycles"]
        if conf["predict_perf_diff"]:
            if conf["predict_perf_direction_additive"]:
                cycles_prediction = wg_cycles_reference["mini_cycles"] + cycles_prediction
            else:
                cycles_prediction = wg_cycles_reference["mini_cycles"] - cycles_prediction
        if cycles_prediction < wg_cycles_reference["rw_cycles"]:
            cycles_prediction = wg_cycles_reference["rw_cycles"]
        error = cycles_prediction - cycles_correct
        header += ",error"
        data_line += ",{0}".format(error)
        error_pct = error / cycles_correct
        header += ",error_pct"
        data_line += ",{0}".format(error_pct)

    with open(filepath, "w") as outfile:
        outfile.write(header + "\n")
        outfile.write(data_line + "\n")

def load_coefficients():
    filepath = os.path.join("Modelling", "solution.csv")
    if not os.path.exists(filepath):
        raise IOError("Cannot find solution file: " + filepath)

    coefs = {}
    with open(filepath, "r") as solution_file:
        csv_reader = csv.reader(solution_file)
        for line in csv_reader:
            if line[1] == "cpi":
                pass
            else:
                coef_name = line[0]
                cpi = float(line[1])
                coefs[coef_name] = cpi

    return coefs

def generate_prediction():
    coef_names = A.columns.values.tolist() + get_meta_coef_names(conf)
    coefs = []
    for c in coef_names:
        coefs.append(solution[c])

    y_model = am.apply_model(coefs)

if __name__ == "__main__":
    main(sys.argv[1:])