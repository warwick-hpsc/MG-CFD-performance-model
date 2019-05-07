import os, subprocess, shutil, sys
from sets import Set
import argparse
import pandas as pd

script_dirpath = os.path.dirname(os.path.realpath(__file__))

## Location of the processed CSV files generated by MG-CFD's 'aggregate-output-data.py':
parser = argparse.ArgumentParser()
parser.add_argument('--data-dirpath', required=True, help="Dirpath to MG-CFD runs processed output data")
args = parser.parse_args()
processed_performance_data_dirpath = args.data_dirpath

## Location of directory containing 'train_model.R':
model_src_dirpath = os.path.join(script_dirpath, "../Main")

## Now train the model to estimate CPI rates of CPU execution units:
tdir="Training"
if not os.path.isdir(tdir):
	os.mkdir(tdir)

for f in ["instruction-counts.mean.csv", "LoopNumIters.mean.csv", "PAPI.mean.csv", "Times.mean.csv"]:
	shutil.copyfile(os.path.join(processed_performance_data_dirpath, f), os.path.join(script_dirpath, tdir, f))

os.chdir(tdir)
# subprocess.call(["Rscript", model_src_dirpath+"/train_model.R"])

pdir="Prediction"
if not os.path.isdir(os.path.join(script_dirpath, pdir)):
	os.mkdir(os.path.join(script_dirpath, pdir))
# for f in ["merged_performance_data.csv", "cpi_estimates.csv"]:
for f in ["cpi_estimates.csv"]:
	shutil.copyfile(os.path.join(script_dirpath, tdir, f), os.path.join(script_dirpath, pdir, f))