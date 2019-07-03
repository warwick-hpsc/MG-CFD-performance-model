import pandas as pd
import numpy as np
import re
from sets import Set

class SupportedArchitectures:
  UNKNOWN = 0
  SANDY = 1
  IVY = 2
  HASWELL = 3
  BROADWELL = 4
  SKYLAKE = 5
  KNL = 6
  WESTMERE = 7

exec_unit_instructions = {}
exec_unit_instructions["alu"] = ["loop", "lea", "^j.*", "cmp", "inc", "add", "mov", "movslq", "movap*", "sar", "[v]?movdq.*", "sh[rl]", "nop", "movzbl"]
exec_unit_instructions["simd_alu"] = ["[v]?pand[d]?", "[v]?andp[sd]", "[v]?xorp.*", "vpxor[d]?", "vmovq", "vmovdq.*", "[v]?paddd", "[v]?psubd", "[v]?pmulld", "vpinsrd", "[v]?punpck.*", "vextracti128"]
exec_unit_instructions["simd_shuffle"] = ["[v]?unpck.*", "vinsertf128", "vperm.*", "vpgather[dq]d", "vgather[dq]pd", "pshufd", "vpblendmq", "vpmovsxdq", "vbroadcast.*", "[v]?pmovzx.*", "vzeroupper"]
exec_unit_instructions["fp_add"] = ["[v]?add[sp]d", "[v]?sub[sp]d", "vpsubq", "[v]?dppd", "[v]?max[sp]d", "[v]?min[sp]d", "[v]?comisd"]
exec_unit_instructions["fp_mul"] = ["mulsd", "[v]?mul[sp]d", "vf[n]?m[as].*"]
exec_unit_instructions["fp_div"] = ["[v]?div[sp]d", "[v]?sqrt[sp]d"]
exec_unit_instructions["fp_div_fast"] = ["vrcp.*", "vrsqrt14pd", "vrsqrt28[sp]d"]
exec_unit_instructions["fp_mov"] = ["[v]?movd", "[v]?movsd", "[v]?movup[sd]", "[v]?movhp[sd]", "[v]?movap[sd]"]
exec_unit_instructions["avx512_alu"] = ["vpxorq", "vptestm.*", "kandw", "kandn.*", "knot.*", "kxorw", "kxnorw"]
exec_unit_instructions["avx512_shuffle"] = ["valign[dq]", "vscatter[dq]p[sd]", "vinserti64x4", "vpbroadcastm.*", "vpbroadcast[bwdq]", "kunpckbw"]
exec_unit_instructions["avx512_misc"] = ["vfpclasspd", "vplzcnt[dq]", "vpconflictd", "vpternlog[dq]", "vfixupimm[sp]d", "kmov[wbqd]", "kshiftrw", "vgetexp[sp][sd]", "vgetmant[sp][sd]", "vscalef[sp][sd]"]
exec_units = exec_unit_instructions.keys()

insignificant_instructions = ["push", "pushq", "pop", "popq", "xor", "xorl", "sub", "subq", "retq", "testb", "and"]

class UnknownInstruction(Exception):
  def __init__(self, insn_name, occurence_count):
    message = "No exec unit found for insn '{0}' which occurs {1} times".format(insn_name, occurence_count)
    super(UnknownInstruction, self).__init__(message)

def get_meta_coef_names(conf):
    meta_coef_names = []
    if conf["do_spill_penalty"]:
        meta_coef_names.append("spill_penalty")
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

def map_insn_to_exec_unit(insn):
  for eu in exec_units:
    if insn in exec_unit_instructions[eu]:
      return eu

  for eu in exec_units:
    for eu_insn in exec_unit_instructions[eu]:
      if re.match(eu_insn, insn):
        return eu

  return ""

def cpu_string_to_arch(cpu):
  is_xeon = "Xeon" in cpu
  is_phi = "Phi" in cpu
  
  if is_xeon and is_phi and "7210" in cpu:
    return SupportedArchitectures.KNL
  if is_xeon and "Silver" in cpu:
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

    if insn in insignificant_instructions:
      continue

    counts["insn."+insn] = count
  return counts

def categorise_instructions_tally(tally_filepath):
  print("Categorising instructions in file: " + tally_filepath)

  eu_classes = ["eu."+eu for eu in exec_units]

  counts = {eu:0 for eu in eu_classes}
  counts["mem.loads"] = 0
  counts["mem.stores"] = 0

  tally = pd.read_csv(tally_filepath)

  for idx, row in tally.iterrows():
    insn = row["insn"].lower()
    count = row["count"]

    if insn in insignificant_instructions:
      continue

    if insn == "loads":
      counts["mem.loads"] += count
      continue
    if insn == "stores":
      counts["mem.stores"] += count
      continue

    eu = map_insn_to_exec_unit(insn)
    exec_unit_found = eu != ""
    if not exec_unit_found:
      raise UnknownInstruction(insn, count)
    counts["eu."+eu] += count

  ## Current Intel documentation does not describe how AVX512 instructions are scheduled to 
  ## execution ports, so for now merge with other categories:
  counts["eu.simd_alu"] = counts["eu.simd_alu"] + counts["eu.avx512_alu"]
  del counts["eu.avx512_alu"]
  counts["eu.simd_shuffle"] = counts["eu.simd_shuffle"] + counts["eu.avx512_shuffle"]
  del counts["eu.avx512_shuffle"]
  counts["eu.fp_mov"] = counts["eu.fp_mov"] + counts["eu.avx512_misc"]
  del counts["eu.avx512_misc"]

  ## Further merging of categories for better model fitting:
  counts["eu.fp_mov"] = counts["eu.fp_mov"] + counts["eu.simd_shuffle"]
  del counts["eu.simd_shuffle"]

  return counts

def categorise_aggregated_instructions_tally(tally_filepath):
  print("Categorising aggregated instructions in file: " + tally_filepath)

  eu_classes = ["eu."+eu for eu in exec_units]

  insn_tally = pd.read_csv(tally_filepath)

  insn_colnames = [c for c in insn_tally.columns.values if c.startswith("insn.")]

  eu_tally = insn_tally.copy().drop(insn_colnames, axis=1)
  for euc in eu_classes:
    eu_tally[euc] = 0
  eu_tally["mem.loads"] = 0
  eu_tally["mem.stores"] = 0

  for insn_cn in insn_colnames:
    insn = insn_cn.split('.')[1].lower()
    count = insn_tally[insn_cn]

    if insn in insignificant_instructions:
      continue

    if insn == "loads":
      eu_tally["mem.loads"] += count
      continue
    if insn == "stores":
      eu_tally["mem.stores"] += count
      continue

    eu = map_insn_to_exec_unit(insn)
    exec_unit_found = eu != ""
    if not exec_unit_found:
      raise UnknownInstruction(insn, count)
    eu_tally["eu."+eu] += count

  ## Current Intel documentation does not describe how AVX512 instructions are scheduled to 
  ## execution ports, so for now merge with other categories:
  eu_tally["eu.simd_alu"] = eu_tally["eu.simd_alu"] + eu_tally["eu.avx512_alu"]
  eu_tally = eu_tally.drop("eu.avx512_alu", axis=1)
  eu_tally["eu.simd_shuffle"] = eu_tally["eu.simd_shuffle"] + eu_tally["eu.avx512_shuffle"]
  eu_tally = eu_tally.drop("eu.avx512_shuffle", axis=1)
  eu_tally["eu.fp_mov"] = eu_tally["eu.fp_mov"] + eu_tally["eu.avx512_misc"]
  eu_tally = eu_tally.drop("eu.avx512_misc", axis=1)

  ## Further merging of categories for better model fitting:
  eu_tally["eu.fp_mov"] = eu_tally["eu.fp_mov"] + eu_tally["eu.simd_shuffle"]
  eu_tally = eu_tally.drop("eu.simd_shuffle", axis=1)

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
    mem_cpis_certain_grp = mem_cpis_certain.groupby(id_cats, as_index=False)
    mem_cpis_certain_means = mem_cpis_certain_grp.mean()

    ## For runs where the modelling was unable to determine CPI, bring 
    ## back the filtered-out defaults:
    if Set(id_groups) != Set(mem_cpis_certain_grp.groups.keys()):
      lost_groups = Set(id_groups).difference(Set(mem_cpis_certain_grp.groups.keys()))
      # # print("WARNING: After filtering-out load/store CPIs of 1.0, all data has been removed for these run ids: " + lost_groups.__str__())
      # print("WARNING: After filtering-out load/store CPIs of 1.0, all data has been removed for these run ids:")
      # print("     " + lost_groups.__str__())
      # print("  Restoring the 1.0 CPI estimates for these runs.")
      for lg in lost_groups:
        lost_cpis = mem_cpis.iloc[mem_cpis_grp.groups[lg]]
        lost_cpis_grp = lost_cpis.groupby(id_cats, as_index=False)
        lost_cpis_means = lost_cpis_grp.mean()
        mem_cpis_means = mem_cpis_means.append(lost_cpis_means)

    ## Write back into eu_cpis:
    eu_cpis_agg = eu_cpis_agg.drop(mem_cat, axis=1).merge(mem_cpis_means)
  
  return eu_cpis_agg
