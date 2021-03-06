import pandas as pd
import numpy as np
from pprint import pprint
import re, os, glob, shutil
from sets import Set

script_dirpath = os.path.dirname(os.path.realpath(__file__))

## Dirpath to MG-CFD performance model:
projection_model_dirpath = os.path.join(script_dirpath, "../", "Main")
py_model_filepath = os.path.join(projection_model_dirpath, "model_interface.py")
import imp
imp.load_source('utils', os.path.join(projection_model_dirpath, "Utils.py"))
from utils import *

## Regenerate all intermediate CSV files?
regen = True
# regen = False

## Read in CPI estimates:
cpi_estimates_fp = os.path.join(script_dirpath, "Prediction", "cpi_estimates.csv")
if not os.path.exists(cpi_estimates_fp):
	raise Exception("Cannot find file 'cpi_estimates'. Have you executed train.py?")
cpi_estimates = pd.read_csv(cpi_estimates_fp, keep_default_na=False)
cpi_estimates = split_var_id_column(cpi_estimates)
cpi_estimates = aggregate_across_instruction_sets(cpi_estimates)
if "OpenMP" in cpi_estimates.columns.values:
	cpi_estimates = cpi_estimates[cpi_estimates["OpenMP"]=="Off"]
	cpi_estimates.drop("OpenMP", axis=1, inplace=True)

## Read in performance data:
prepared_performance_data_fp = os.path.join(script_dirpath, "Prediction", "prepared_performance_data.csv")
if not os.path.exists(prepared_performance_data_fp):
	raise Exception("Cannot find file 'prepared_performance_data'. Have you executed prepare_data_for_predict.py?")
mini_perf_data = pd.read_csv(prepared_performance_data_fp)
mini_perf_data = mini_perf_data.rename(index=str, columns={"wg_cycles":"mini_wg_cycles"})

## Read in target kernel data:
# target_eu_counts_fp = os.path.join(script_dirpath, "Prediction", "target_eu_counts.csv")
target_eu_counts_fp = os.path.join(script_dirpath, "target_eu_counts.csv")
if regen or not os.path.isfile(target_eu_counts_fp):
	target_insn_counts_fp = os.path.join(script_dirpath, "target_insn_counts.csv")
	if os.path.isfile(target_insn_counts_fp):
		target_eu_counts = categorise_aggregated_instructions_tally(target_insn_counts_fp)
		if "eu.load" in target_eu_counts.keys():
			## Can ignore load-related instructions (eg vmovsd) as I think they 
			## map directly to just an actual memory load, which I already track.
			del target_eu_counts["eu.load"]
		target_eu_counts.to_csv(target_eu_counts_fp, index=False)
	else:
		if not os.path.isfile(target_eu_counts_fp):
			raise Exception("Need 'target_eu_counts.csv' or 'target_insn_counts.csv'")
		else:
			target_eu_counts = pd.read_csv(target_eu_counts_fp)
else:
	target_eu_counts = pd.read_csv(target_eu_counts_fp)

##################################
## Prepare data for prediction ...
##################################

cpu_values = Set(mini_perf_data["CPU"])
if len(cpu_values) > 1:
	raise Exception("Source performance data contains multiple CPU values, expected just one.")
cpu = mini_perf_data["CPU"][0]
target_arch = cpu_string_to_arch(cpu)


## Prepare kernel instruction counts:
insn_cats = [i for i in target_eu_counts.columns.values if ("eu." in i or "mem." in i)]
mgcfd_eu_counts_colnames = list(Set(mini_perf_data.columns.values).intersection(target_eu_counts.columns.values))
f = np.logical_and(mini_perf_data["level"]==0, mini_perf_data["kernel"]=="flux")
mgcfd_eu_counts = mini_perf_data[f]
if "Num.threads" in mgcfd_eu_counts.columns.values:
	mgcfd_eu_counts = mgcfd_eu_counts[mgcfd_eu_counts["Num.threads"]==1]
	mgcfd_eu_counts.drop("Num.threads", axis=1, inplace=True)
mgcfd_eu_counts  =  mgcfd_eu_counts[mgcfd_eu_counts_colnames]
mgcfd_eu_counts  =  mgcfd_eu_counts.rename(index=str, columns={i:i+".mgcfd"  for i in insn_cats})
target_eu_counts = target_eu_counts.rename(index=str, columns={i:i+".target" for i in insn_cats})

## Prepare empirical performance data:
mini_wg_cycles_columns = Set(["Instruction.set", "Precise.FP", "Num.threads", "kernel", "level", "mini_wg_cycles"]).intersection(mini_perf_data.columns.values)
mini_wg_cycles = mini_perf_data[list(mini_wg_cycles_columns)]
mini_wg_cycles = mini_wg_cycles.pivot_table(index=list(mini_wg_cycles_columns.difference(["kernel", "mini_wg_cycles"])), columns="kernel", values="mini_wg_cycles").reset_index()
mini_wg_cycles = mini_wg_cycles.rename(index=str, columns={"flux":"mini_wg_cycles"})
mini_wg_cycles = mini_wg_cycles.rename(index=str, columns={"indirect_rw":"mini_wg_cycles_rw"})
flux_wg_cycles = mini_wg_cycles.copy()
if "Num.threads" in flux_wg_cycles.columns.values:
	flux_wg_cycles = flux_wg_cycles[flux_wg_cycles["Num.threads"]==1].drop(["Num.threads"], axis=1)
flux_wg_cycles = flux_wg_cycles.drop("mini_wg_cycles_rw", axis=1)

ghz_data_columns = Set(["Instruction.set", "Precise.FP", "level", "Num.threads", "kernel", "GHz"]).intersection(mini_perf_data.columns.values)
ghz_data = mini_perf_data[ghz_data_columns]
kernels = Set(ghz_data["kernel"])
# ghz_data = ghz_data.pivot_table(index=["Instruction.set", "Num.threads"], columns="kernel", values="GHz").reset_index()
ghz_data = ghz_data.pivot_table(index=list(ghz_data_columns.difference(["kernel", "GHz"])), columns="kernel", values="GHz").reset_index()
ghz_data = ghz_data.rename(index=str, columns={s:"ghz_"+s for s in kernels})


## Merge together data to form prediction dataset:
input_prediction_data = flux_wg_cycles.merge(cpi_estimates)
if input_prediction_data.shape[0]==0:
	raise Exception("Merge of <mini_wg_cycles, cpi_estimates> produced empty DataFrame")
if "Num.threads" in input_prediction_data.columns.values:
	input_prediction_data = input_prediction_data[input_prediction_data["Num.threads"]==1]
	input_prediction_data.drop("Num.threads", axis=1, inplace=True)
input_prediction_data = input_prediction_data.rename(index=str, columns={i:i+".cpi" for i in insn_cats})
eu_cpi_col_names = [i+".cpi" for i in insn_cats]

input_prediction_data = input_prediction_data.merge(target_eu_counts)
input_prediction_data = input_prediction_data.merge(mgcfd_eu_counts)
input_prediction_data_fp = os.path.join(script_dirpath, "Prediction", "input_prediction_data.csv")
input_prediction_data.to_csv(input_prediction_data_fp, index=False)


def get_id_colnames(df):
	known_data_colnames = ["target_more_expensive"]
	cn = df.columns.values
	cn = [c for c in cn if not "eu." in c]
	cn = [c for c in cn if not "mem." in c]
	cn = [c for c in cn if not "wg" in c]
	cn = list(Set(cn).difference(known_data_colnames))
	return cn

def generate_wg_cycles_predictions(input_prediction_data):
	if not os.path.isdir("Modelling"):
		os.mkdir("Modelling")

	target_predictions = [0] * input_prediction_data.shape[0]

	for idx in range(input_prediction_data.shape[0]):
		print("Predicting {0} of {1}".format(idx+1, input_prediction_data.shape[0]))

		model_data = input_prediction_data.iloc[idx]

		insn_cats_nonzero = [insn_cat for insn_cat in insn_cats if model_data[insn_cat+".target"] > 0]

		## Write out target instruction counts:
		pred_data_fp = os.path.join("Modelling", "prediction_data.csv")
		header = "wg_cycles"
		data_line = "0"
		for insn_cat in insn_cats_nonzero:
			if header != "":
				header += ","
				data_line += ","
			header += insn_cat
			data_line += "{0}".format(model_data[insn_cat+".target"])
		with open (pred_data_fp, "w") as pred_data_file:
			pred_data_file.write(header+"\n")
			pred_data_file.write(data_line+"\n")

		## Write out MG-CFD as calibration data:
		calib_data_fp = os.path.join("Modelling", "calibration_data.csv")
		header = "wg_cycles"
		data_line = "{0}".format(model_data["mini_wg_cycles"])
		header += ',' + ','.join(insn_cats_nonzero)
		data_line += ',' + ','.join(["{0}".format(model_data[i+".mgcfd"]) for i in insn_cats_nonzero])
		with open (calib_data_fp, "w") as calib_data_file:
			calib_data_file.write(header+"\n")
			calib_data_file.write(data_line+"\n")
		header = "wg_cycles" + "," + ",".join(insn_cats_nonzero)

		## Write out CPI estimates:
		sol_data_fp = os.path.join("Modelling", "solution.csv")
		with open(sol_data_fp, "w") as sol_data_file:
			sol_data_file.write("coef,cpi\n")
			for insn_cat in insn_cats_nonzero:
				try:
					cpi = model_data[insn_cat+".cpi"]
				except:
					cpi = 1.0
				sol_data_file.write("{0},{1}\n".format(insn_cat, cpi))

		arch_to_flag = {
			SupportedArchitectures.SANDY: "cpu_is_sandy",
			SupportedArchitectures.IVY: "cpu_is_ivy",
			SupportedArchitectures.HASWELL: "cpu_is_haswell", 
			SupportedArchitectures.BROADWELL: "cpu_is_broadwell",
			SupportedArchitectures.SKYLAKE: "cpu_is_skylake",
			SupportedArchitectures.CASCADELAKE: "cpu_is_skylake",
			SupportedArchitectures.KNL: "cpu_is_knl",
			SupportedArchitectures.WESTMERE: "cpu_is_westmere"
		}

		## Write out model config:
		model_conf_fp = os.path.join("Modelling", "insn_model_conf.csv")
		with open(model_conf_fp, "w") as model_conf_file:
			model_conf_file.write("key,value\n")
			model_conf_file.write(arch_to_flag[target_arch] + ",TRUE\n")

			model_conf_file.write("predict_perf_diff" + ",TRUE\n")
			## Target is more expensive than MG-CFD:
			model_conf_file.write("predict_perf_direction_additive" + ",TRUE\n")

		## Run model:
		os.system("python {0} -p".format(py_model_filepath))

		model_prediction_fp = os.path.join("Modelling", "prediction.csv")
		if not os.path.isfile(model_prediction_fp):
			raise Exception("model did not output prediction.")

		## Read in prediction
		model_input_prediction_data = pd.read_csv(model_prediction_fp)
		cycles_prediction = model_input_prediction_data.iloc[0]["cycles_model"]
		os.remove(model_prediction_fp)

		target_wg_prediction = cycles_prediction
		if target_wg_prediction < 0.0:
			raise Exception("target_wg_prediction is negative: {0}".format(target_wg_prediction))
		target_predictions[idx] = target_wg_prediction

	## Cleanup:
	if os.path.isdir("Modelling"):
		shutil.rmtree("Modelling")

	predictions_df = pd.DataFrame(input_prediction_data[get_id_colnames(input_prediction_data)])
	predictions_df["target_wg_cycles_prediction"] = target_predictions

	# ## Prune non-varying columns:
	# for c in predictions_df.columns.values:
	# 	if len(predictions_df[c].unique()) == 1:
	# 		predictions_df.drop(c, axis=1, inplace=True)

	return predictions_df

target_wg_predictions_filepath = os.path.join(script_dirpath, "Prediction", "target_wg_predictions.csv")
if regen or not os.path.isfile(target_wg_predictions_filepath):
	wg_cycles_predictions = generate_wg_cycles_predictions(input_prediction_data)

	## Now prepare a multi-core oriented dataset for generating scaling predictions:
	mc_prediction_data = mini_wg_cycles.merge(ghz_data, validate="many_to_one")
	mc_prediction_data["wg_nsec_rw"] = mc_prediction_data["mini_wg_cycles_rw"] / mc_prediction_data["ghz_indirect_rw"]

	mc_prediction_data = mc_prediction_data.merge(wg_cycles_predictions, validate="many_to_one")
	mc_prediction_data["target_wg_nsec_prediction"] = mc_prediction_data["target_wg_cycles_prediction"] / mc_prediction_data["ghz_flux"]
	mc_prediction_data["target_wg_nsec_prediction"] = np.maximum(mc_prediction_data["target_wg_nsec_prediction"], mc_prediction_data["wg_nsec_rw"])
	mc_prediction_data["mini_wg"] = mc_prediction_data["mini_wg_cycles"] / mc_prediction_data["ghz_flux"] / 1e9

	## Clean:
	# mc_prediction_data = mc_prediction_data.drop("target_wg_cycles_prediction", axis=1)
	# mc_prediction_data = mc_prediction_data.drop(["ghz_flux"], axis=1)
	mc_prediction_data = mc_prediction_data.drop(["ghz_indirect_rw"], axis=1)
	# mc_prediction_data = mc_prediction_data.drop(["mini_wg_cycles", "mini_wg_cycles_rw"], axis=1)
	mc_prediction_data = mc_prediction_data.drop(["wg_nsec_rw"], axis=1)
	mc_prediction_data["target_wg_prediction"] = mc_prediction_data["target_wg_nsec_prediction"] / 1e9
	mc_prediction_data = mc_prediction_data.drop("target_wg_nsec_prediction", axis=1)

	print("Writing target WG prediction(s) to: " + target_wg_predictions_filepath)
	mc_prediction_data.to_csv(target_wg_predictions_filepath, index=False)
else:
	mc_prediction_data = pd.read_csv(target_wg_predictions_filepath)
