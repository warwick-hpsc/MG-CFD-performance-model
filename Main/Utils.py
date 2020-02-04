import pandas as pd
import numpy as np
import re, os
from sets import Set

from pprint import pprint

utils_script_dirpath = os.path.dirname(os.path.realpath(__file__))

class SupportedArchitectures:
  UNKNOWN = 0
  SANDY = 1
  IVY = 2
  HASWELL = 3
  BROADWELL = 4
  SKYLAKE = 5
  CASCADELAKE = 6
  KNL = 7
  WESTMERE = 8

class UnknownInstruction(Exception):
  def __init__(self, insn_name, occurence_count):
    message = "No exec unit found for insn '{0}' which occurs {1} times".format(insn_name, occurence_count)
    super(UnknownInstruction, self).__init__(message)

def safe_pd_filter(df, field, value):
  if not field in df.columns.values:
    print("WARNING: field '{0}' not in df".format(field))
    return df

  if isinstance(value, list):
    if len(value) == 0:
      raise Exception("safe_pd_filter() passed an empty list of values")
    else:
      f = df[field]==value[0]
      for i in range(1,len(value)):
        f = np.logical_or(f, df[field]==value[i])
      df = df[f]
  else:
    df = df[df[field]==value]

  if len(Set(df[field])) == 1:
    df = df.drop(field, axis=1)

  nrows = df.shape[0]
  if nrows == 0:
    raise Exception("No rows left after filter: '{0}' == '{1}'".format(field, value))
  return df

def load_insn_eu_mapping():
  exec_unit_mapping_filepath = os.path.join(utils_script_dirpath, "Backend", "insn_eu_mapping.csv")
  df = pd.read_csv(exec_unit_mapping_filepath)

  exec_unit_mapping = {}
  for index,row in df.iterrows():
    eu = row["exec_unit"]
    if not eu in exec_unit_mapping:
      exec_unit_mapping[eu] = [row["instruction"]]
    else:
      exec_unit_mapping[eu].append(row["instruction"])

  return exec_unit_mapping

def get_meta_coef_names(conf):
    meta_coef_names = []
    if conf["do_spill_penalty"]:
        meta_coef_names.append("spill_penalty")
    if conf["do_load_penalty"]:
        meta_coef_names.append("load_penalty")
    return meta_coef_names

def get_meta_coefs(conf, coefs):
    meta_coefs = {}

    meta_coef_names = get_meta_coef_names(conf)

    num_insn_coefs = len(coefs) - len(meta_coef_names)

    ni = num_insn_coefs
    num_coefs = ni + len(meta_coef_names)
    for i in range(ni, num_coefs):
    	meta_coefs[meta_coef_names[i-ni]] = coefs[i]

    return meta_coefs

def split_var_id_column(df):
  if not "var_id" in df.columns.values:
    return df

  var_id_values = df["var_id"]
  var_id_split = var_id_values.str.split('^', expand=True)
  for col in var_id_split.columns.values:
    var_id_split_col = var_id_split[col].str.split('=', expand=True)
    var_name = var_id_split_col.iloc[0,0]
    var_values = var_id_split_col.iloc[:,1]

    if var_name == "level" or var_name == "SIMD.len":
      var_values = var_values.astype(np.int)

    df[var_name] = var_values
  df = df.drop("var_id", axis=1)
  return df

def map_insn_to_exec_unit(insn, mapping):
  exec_units = mapping.keys()
  for eu in exec_units:
    if insn in mapping[eu]:
      return eu

  for eu in exec_units:
    for eu_insn in mapping[eu]:
      if re.match(eu_insn, insn):
        return eu

  return ""

def cpu_string_to_arch(cpu):
  is_xeon = "Xeon" in cpu
  is_phi = "Phi" in cpu
  
  if is_xeon and is_phi and "7210" in cpu:
    return SupportedArchitectures.KNL
  if is_xeon and re.search("[0-9]2[0-9][0-9]", cpu):
    return SupportedArchitectures.CASCADELAKE
  if is_xeon and re.search("[0-9]1[0-9][0-9]", cpu):
    return SupportedArchitectures.SKYLAKE
  if is_xeon and "v4" in cpu:
    return SupportedArchitectures.BROADWELL
  if "i5-4" in cpu:
    return SupportedArchitectures.HASWELL
  if is_xeon and "v2" in cpu:
    return SupportedArchitectures.IVY
  if "i5-2" in cpu:
    return SupportedArchitectures.SANDY
  if "X5650" in cpu:
    return SupportedArchitectures.WESTMERE

  raise Exception("Do not know arch of CPU string '{0}'".format(cpu))

def instructions_tally_to_dict(tally_filepath):
  counts = {}
  tally = pd.read_csv(tally_filepath)
  for idx, row in tally.iterrows():
    insn = row["insn"].lower()
    count = row["count"]

    counts["insn."+insn] = count
  return counts

def categorise_instructions_tally(tally_filepath):
  # print("Categorising instructions in file: " + tally_filepath)

  eu_mapping = load_insn_eu_mapping()
  exec_units = eu_mapping.keys()
  eu_classes = ["eu."+eu for eu in exec_units]

  counts = {eu:0 for eu in eu_classes}
  counts["mem.loads"] = 0
  counts["mem.stores"] = 0
  counts["mem.load_spills"] = 0
  counts["mem.store_spills"] = 0

  tally = pd.read_csv(tally_filepath)

  for idx, row in tally.iterrows():
    insn = row["insn"].lower()
    count = row["count"]

    if insn == "loads":
      counts["mem.loads"] += count
      continue
    elif insn == "stores":
      counts["mem.stores"] += count
      continue
    elif insn == "load_spills":
      counts["mem.load_spills"] += count
      continue
    elif insn == "store_spills":
      counts["mem.store_spills"] += count
      continue

    eu = map_insn_to_exec_unit(insn, eu_mapping)
    if eu == "":
      raise UnknownInstruction(insn, count)
    counts["eu."+eu] += count

  if "eu.DISCARD" in counts.keys():
    del counts["eu.DISCARD"]

  return counts

def categorise_aggregated_instructions_tally(tally_filepath):
  # print("Categorising aggregated instructions in file: " + tally_filepath)

  eu_mapping = load_insn_eu_mapping()
  exec_units = eu_mapping.keys()
  eu_classes = ["eu."+eu for eu in exec_units]

  insn_tally = pd.read_csv(tally_filepath, keep_default_na=False)

  insn_colnames = [c for c in insn_tally.columns.values if c.startswith("insn.")]

  eu_tally = insn_tally.copy().drop(insn_colnames, axis=1)
  for euc in eu_classes:
    eu_tally[euc] = 0
  eu_tally["mem.loads"] = 0
  eu_tally["mem.stores"] = 0
  # eu_tally["mem.spills"] = 0
  eu_tally["mem.load_spills"] = 0
  eu_tally["mem.store_spills"] = 0

  for insn_cn in insn_colnames:
    insn = insn_cn.split('.')[1].lower()
    count = insn_tally[insn_cn]

    if insn == "loads":
      eu_tally["mem.loads"] += count
      continue
    elif insn == "stores":
      eu_tally["mem.stores"] += count
      continue
    elif insn == "load_spills":
      eu_tally["mem.load_spills"] += count
      continue
    elif insn == "store_spills":
      eu_tally["mem.store_spills"] += count
      continue

    eu = map_insn_to_exec_unit(insn, eu_mapping)
    exec_unit_found = eu != ""
    if not exec_unit_found:
      raise UnknownInstruction(insn, count.values.max())
    eu_tally["eu."+eu] += count

  if "eu.DISCARD" in eu_tally.keys():
    del eu_tally["eu.DISCARD"]

  if "kernel" in eu_tally.columns.values:
    if "compute_flux_edge" in Set(eu_tally["kernel"]) and "indirect_rw" in Set(eu_tally["kernel"]):
      ## Good, have enough data to distinguish between spill-induced L1 loads/stores and main memory loads/stores. 
      ## Can address situations where assembly-loop-extractor failed to identify spills:
      rw_data = safe_pd_filter(eu_tally, "kernel", "indirect_rw")
      if rw_data.shape[0] == eu_tally[eu_tally["kernel"]=="compute_flux_edge"].shape[0]:
        ## Safe to merge:
        rw_data = rw_data.drop(columns=[c for c in rw_data.columns if c.startswith("eu.")])
        rw_data = rw_data.rename(columns={c:c+".rw" for c in rw_data.columns if c.startswith("mem.")})
        eu_tally = eu_tally.merge(rw_data)
        f = eu_tally["mem.load_spills"]==0
        eu_tally.loc[f,"mem.load_spills"] = eu_tally.loc[f,"mem.loads"] - eu_tally.loc[f,"mem.loads.rw"]
        eu_tally.loc[f,"mem.loads"] = eu_tally.loc[f,"mem.loads.rw"]
        f = eu_tally["mem.store_spills"]==0
        eu_tally.loc[f,"mem.store_spills"] = eu_tally.loc[f,"mem.stores"] - eu_tally.loc[f,"mem.stores.rw"]
        eu_tally.loc[f,"mem.stores"] = eu_tally.loc[f,"mem.stores.rw"]
        eu_tally = eu_tally.drop(columns=[c for c in eu_tally.columns if c.endswith(".rw")])

  return eu_tally

def aggregate_across_instruction_sets(eu_cpis):
  ## This function is initially focused on performance of loads and stores, which is 
  ## considered independent of the instruction set used. However, the ability to 
  ## measure the performance is sensitive to instruction set; measurement is easier 
  ## with older isets which put greater pressure on L1 R/W.

  iset_colname_candiates = ["iset", "Instruction.set"]
  iset_colname = ""
  for c in iset_colname_candiates:
    if c in eu_cpis.columns.values:
      iset_colname = c
      break
  if iset_colname == "":
    print("WARNING: aggregate_across_instruction_sets() called on DataFrame with no 'iset' column, so nothing to do.")
    return eu_cpis

  eu_cpis_agg = eu_cpis.copy()

  eu_cats  = [x for x in eu_cpis.columns.values if "eu." in x]
  mem_cats = [x for x in eu_cpis.columns.values if "mem." in x]
  id_cats = list(Set(eu_cpis.columns.values).difference(mem_cats).difference(eu_cats).difference([iset_colname]))

  ## For each 'mem_cat', perform an aggregation across 
  ## each element of 'id_cats'. 
  ## If model training successfully estimated CPI values, 
  ## then > 1.0 values will be present - average across just those. 
  ## If unsuccessful, then model will have selected the 
  ## lower bound value of 1.0, so just return that.
  for mem_cat in mem_cats:
    if len(id_cats) == 0:
      mem_cpis = eu_cpis_agg[mem_cat].copy()
      if sum(mem_cpis != 1.0) > 0:
        mem_cpis = mem_cpis[mem_cpis != 1.0]
      mem_cpis_mean = mem_cpis.mean()
      eu_cpis_agg = eu_cpis_agg.drop(mem_cat, axis=1)
      eu_cpis_agg[mem_cat] = mem_cpis_mean
      continue

    mem_cpis = eu_cpis[id_cats + [mem_cat]]
    mem_cpis_grp = mem_cpis.groupby(id_cats, as_index=False)
    id_groups = mem_cpis_grp.groups.keys()

    mem_cpis_certain = mem_cpis[mem_cpis[mem_cat] != 1.0]
    if mem_cpis_certain.shape[0] == 0:
      mem_cpis_means = None
    else:
      mem_cpis_certain_grp = mem_cpis_certain.groupby(id_cats, as_index=False)
      mem_cpis_means = mem_cpis_certain_grp.mean()

    ## For runs where the modelling was unable to determine CPI, bring 
    ## back the filtered-out defaults:
    lost_groups = Set(id_groups)
    if not mem_cpis_means is None:
      lost_groups = lost_groups.difference(Set(mem_cpis_certain_grp.groups.keys()))
    if len(lost_groups) > 0:
      # # print("WARNING: After filtering-out load/store CPIs of 1.0, all data has been removed for these run ids: " + lost_groups.__str__())
      # print("WARNING: After filtering-out load/store CPIs of 1.0, all data has been removed for these run ids:")
      # print("     " + lost_groups.__str__())
      # print("  Restoring the 1.0 CPI estimates for these runs.")
      for lg in lost_groups:
        lost_cpis = mem_cpis.iloc[mem_cpis_grp.groups[lg]]
        lost_cpis_grp = lost_cpis.groupby(id_cats, as_index=False)
        lost_cpis_means = lost_cpis_grp.mean()
        if mem_cpis_means is None:
          mem_cpis_means = lost_cpis_means
        else:
          mem_cpis_means = mem_cpis_means.append(lost_cpis_means)

    ## Write back into eu_cpis:
    eu_cpis_agg = eu_cpis_agg.drop(mem_cat, axis=1).merge(mem_cpis_means)
    if eu_cpis_agg.shape[0]==0:
      print(mem_cpis_means)
      raise Exception("Merge of eu_cpis_agg with mem_cpis_means failed")
  
  return eu_cpis_agg

def pd_cartesian_merge(df1, df2):
  merge_key = "_tmp"
  while merge_key in df1.columns.values and merge_key in df2.columns.values:
    merge_key += "x"

  df1[merge_key] = 1
  df2[merge_key] = 1
  df = pd.merge(df1, df2, on=merge_key).drop(merge_key, axis=1)
  # df.index = pd.MultiIndex.from_product((df1.index, df2.index))
  df1.drop(merge_key, axis=1, inplace=True)
  df2.drop(merge_key, axis=1, inplace=True)
  return df
