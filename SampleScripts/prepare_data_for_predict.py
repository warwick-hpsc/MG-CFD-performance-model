import pandas as pd
import numpy as np
from pprint import pprint
import re, os, glob, shutil
from sets import Set

script_dirpath = os.path.dirname(os.path.realpath(__file__))

in_data_filepath = os.path.join(script_dirpath, "Training", "merged_performance_data.csv")
out_data_filepath = os.path.join(script_dirpath, "Prediction", "prepared_performance_data.csv")

perf_data = pd.read_csv(in_data_filepath)

perf_data.columns = perf_data.columns.str.replace(' ', '.')
filters = {}
filters["Flux.options"] = ""
filters["Flux.variant"] = "Normal"
filters["Flux.fission"] = 'N'
filters["OpenMP"] = 'Strong'
for c in filters.keys():
	if c in perf_data.columns.values:
		v = filters[c]
		perf_data_filtered = perf_data[perf_data[c]==v].drop(c, axis=1)
		if perf_data_filtered.shape[0] == 0:
			raise Exception("No rows left after filtering 'instruction-counts.mean.csv' on '{0}' == '{1}'.".format(c, v))
		perf_data = perf_data_filtered
mandatory_columns = ["Instruction.set", "CPU", "niters"]
for c in perf_data.columns.values:
	if c in mandatory_columns:
		continue
	elif "insn." in c or "eu." in c or "mem." in c:
		continue
	elif len(Set(perf_data[c])) == 1:
		perf_data = perf_data.drop(c, axis=1)

perf_data["GHz"] = perf_data["PAPI_TOT_CYC_MAX"] / 1e9 / perf_data["runtime"]
perf_data["wg_cycles"] = perf_data["PAPI_TOT_CYC_MAX"] / perf_data["niters"]
perf_data["wg"] = perf_data["wg_cycles"] / (perf_data["GHz"] * 1e9)

perf_data.to_csv(out_data_filepath, index=False)
