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

input_dirname = "Modelling"

y_name = "wg_cycles"
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='fit', action='store_true')
    parser.add_argument('-p', help='predict', action='store_true')
    parser.add_argument('-d', help="Dirname")
    args = parser.parse_args()

    global input_dirname
    if not args.d is None:
        input_dirname = args.d

    init_np()

    conf = init_conf()

    if args.f:
        y, A = load_fitting_data()

        if "do_prune_insn_classes" in conf and conf["do_prune_insn_classes"]:
            # Drop columns with low value counts, insufficient data for CPI estimation 
            # and leads to model over-fitting to these columns.
            A_colnames = A.columns.values
            # threshold = 5.0
            threshold = 6.0
            fast_insn_classes = ["eu.alu", "eu.simd_alu", "eu.fp_add", "eu.fp_mul"]
            for i in fast_insn_classes:
                if i in A.columns.values and A[i].mean() < threshold:
                    A = A.drop(i, axis=1)

        am = ArchModel(conf, A)
        solution = find_solution(conf, A, y, am)
        write_solution(solution)

    elif args.p:

        coefs = load_coefficients()

        if not conf["predict_perf_diff"]:
            y, A = load_calibration_data()
            am = ArchModel(conf, A)
            y_predict = am.predict(coefs, do_print=False, return_bottleneck=False)
            y_predict = y_predict[0]
            idle_cycles = max(0.0, y[0] - y_predict)
            # print("Calibration idle_cycles = {0}".format(idle_cycles))
        else:
            idle_cycles = 0.0

        y, A = load_predict_data()
        am = ArchModel(conf, A)
        # y_predict = am.predict(coefs)
        y_predict, bottleneck = am.predict(coefs, do_print=False, return_bottleneck=True)
        y_predict = y_predict[0]
        if idle_cycles != 0.0:
            if bottleneck is None or bottleneck == "":
                bottleneck = "idle_cycles={0}".format(round(idle_cycles))
            else:
                bottleneck += ";idle_cycles={0}".format(round(idle_cycles))

        if not conf["predict_perf_diff"]:
            y_predict += idle_cycles

        wg_cycles_reference = load_validation_data()
        if wg_cycles_reference != None:
            y_predict = max(y_predict, wg_cycles_reference["rw_cycles"])
        
        write_prediction(conf, y_predict, bottleneck)

def init_np():
    float_formatter = lambda x: "%+.2E" % x
    # float_formatter = lambda x: "%+.1E" % x
    # float_formatter = lambda x: "%.3f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter}, linewidth=np.nan)

    # np.random.seed(123456)
    np.random.seed(65432)

def init_conf():
    conf_filepath = os.path.join(input_dirname, "insn_model_conf.csv")

    ## Model config:
    conf = {}
    conf["do_spill_penalty"] = False
    conf["do_load_penalty"] = False
    conf["do_prune_insn_classes"] = False
    conf["do_ignore_loads_stores"] = False
    conf["cpu_is_skylake"] = False
    conf["cpu_is_broadwell"] = False
    conf["cpu_is_haswell"] = False
    conf["cpu_is_ivy"] = False
    conf["cpu_is_sandy"] = False
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
    # data = data.drop_duplicates(coef_names)
    ## Instead, use average across duplicate rows:
    data_grp = data.groupby(coef_names)
    data_mean = data_grp.mean().reset_index()

    y = data[y_name]
    A = data[coef_names]

    return y, A

def load_fitting_data():
    fitting_data_filepath = os.path.join(input_dirname, "fitting_data.csv")
    return load_data(fitting_data_filepath)

def load_predict_data():
    filepath = os.path.join(input_dirname, "prediction_data.csv")
    return load_data(filepath)

def load_calibration_data():
    filepath = os.path.join(input_dirname, "calibration_data.csv")
    return load_data(filepath)

def load_validation_data():
    filepath = os.path.join(input_dirname, "validate_prediction_data.csv")
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
    y_error = y_model-y.values
    y_error_pct = np.divide(y_error, y)
    print("y_error_pct:")
    print(["{0}%".format(round(100.0*p, 1)) for p in y_error_pct])
    # y_error_sum = s.calc_model_error_sum(coef_final_estimate)
    # print("y_error_sum = {0}".format(y_error_sum))

    coef_names = s.get_coef_names()
    solution_dict = {}
    for i in range(len(coef_names)):
        solution_dict[coef_names[i]] = coef_final_estimate[i]

    sum_error_pct = np.absolute(y_error).sum() / y.values.sum()
    solution_dict["sum_error_pct"] = sum_error_pct

    wc_idx = np.argmax(np.absolute(y_error))
    wc_error_pct = abs(y_error[wc_idx] / y.values[wc_idx])
    solution_dict["worst_error_pct"] = wc_error_pct

    return solution_dict


def write_solution(coef_final_estimate):
    solution_filepath = os.path.join(input_dirname, "solution.csv")
    if os.path.exists(solution_filepath):
        os.remove(solution_filepath)

    with open(solution_filepath, "w") as solution_file:
        solution_file.write("coef,cpi\n")
        for k in coef_final_estimate:
            # cpi = coef_final_estimate[k]
            cpi = round(coef_final_estimate[k], 4)
            solution_file.write("{0},{1}\n".format(k, cpi))

def write_prediction(conf, cycles_prediction, bottleneck=None):
    filepath = os.path.join(input_dirname, "prediction.csv")
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

    if not bottleneck is None:
        header += ",bottleneck"
        data_line += ",{0}".format(bottleneck)

    with open(filepath, "w") as outfile:
        outfile.write(header + "\n")
        outfile.write(data_line + "\n")

def load_coefficients():
    filepath = os.path.join(input_dirname, "solution.csv")
    if not os.path.exists(filepath):
        raise IOError("Cannot find solution file: " + filepath)

    coefs = {}
    with open(filepath, "r") as solution_file:
        csv_reader = csv.reader(solution_file)
        for line in csv_reader:
            if line[1] == "cpi":
                # Header line
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
