import pandas as pd
import os
import argparse

script_dirpath = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--assembly-dirpath',       required=True,  help="Dirpath to assembly-loop-extractor")
parser.add_argument('--target-object-filepath', required=True,  help="Filepath to object file containing target loop")
parser.add_argument('--object-is-target-loop',  required=False, action='store_true', help='Is object file of target loop alone in a single function?')
parser.add_argument('--target-loop-name',       required=False, help="If object contains much more than just target loop, then must specify "
                                                                     "name of function containing target loop. This is to focus extractor to "
                                                                     "the correct loop. Note that extraction this way is experimental")
parser.add_argument('--instruction-set',        required=True,  help="Instruction set of target object. Set to one of: Host, SSE41, SSE42, AVX, AVX2, or AVX512")
args = parser.parse_args()
if not args.object_is_target_loop:
	if args.target_loop_name is None:
		raise Exception("'object_is_target_loop' is False and 'target_loop_name' not set. If object file contains much more than target loop, then must specify name of function containing the loop.")
	else:
		target_loop_name = args.target_loop_name
target_object_filepath = args.target_object_filepath

iset = args.instruction_set

# out_dir = "Data"
out_dir = "./"

import imp
imp.load_source('assembly_analysis', os.path.join(args.assembly_dirpath, "assembly_analysis.py"))
from assembly_analysis import *
imp.load_source('utils', os.path.join(script_dirpath, "../Main/Utils.py"))
from utils import *

def generate_target_loop_tally(obj_filepath):
	asm_filepath = obj_to_asm(obj_filepath)

	## If object file *only* contains the target loop, then there 
	## is no need to parse the assembly for the loop, can just 
	## treat the whole object as the loop.
	object_is_target_loop = True
	# object_is_target_loop = False

	if object_is_target_loop:
		asm = AssemblyObject(asm_filepath, "")
	else:
		asm = AssemblyObject(asm_filepath, target_loop_name)
	
	asm = AssemblyObject(asm_filepath, "")
	asm.write_out_asm_simple()
	loop = Loop(0, len(asm.operations)-1)
	loop.unroll_factor = 1
	loop_stats = count_loop_instructions(asm_filepath+".simple", loop)
	loop_stats_filepath = asm_filepath + ".stats.csv"
	with open(loop_stats_filepath, "w") as outfile:
	  outfile.write("insn,count\n")
	  for insn in loop_stats.keys():
		outfile.write("{0},{1}\n".format(insn, loop_stats[insn]))

out_filename = "target_insn_counts.csv"
out_filepath = os.path.join(out_dir, out_filename)
# if not os.path.isfile(out_filepath):
asm_tally_filepath = target_object_filepath + ".asm.stats.csv"
if not os.path.isfile(asm_tally_filepath):
	generate_target_loop_tally(target_object_filepath)

insn_tally = instructions_tally_to_dict(asm_tally_filepath)

insn_tally_df = pd.DataFrame(data=insn_tally, index=[0])
insn_tally_df["iset"] = iset

if not os.path.isdir(out_dir):
	os.mkdir(out_dir)
insn_tally_df.to_csv(out_filepath, index=False)
print("Target insn counts written to file: {0}".format(out_filepath))
