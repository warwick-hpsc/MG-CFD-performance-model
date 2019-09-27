import numpy as np
import os, sys

from pprint import pprint

script_dirpath = os.path.dirname(os.path.realpath(__file__))

import imp
imp.load_source('Utils', os.path.join(script_dirpath, "../Utils.py"))
from Utils import *

class ArchModel(object):
    def __init__(self, conf, A):
        self.conf = conf

        # Update: treat spills as special case, using same CPI as mem.loads:
        if "mem.spills" in A.columns.values:
            # Move to the end of frame to enable special treatment:
            A = A[[c for c in A.columns.values if c != "mem.spills"] + ["mem.spills"]]

        if conf["cpu_is_skylake"]:
            self.fp_add_ports = [0,1]
            self.fp_mul_ports = [0,1]
            self.avx512_ports = [0,1]
            self.vec_alu_ports = [0,1,5]
            self.fp_shuf_ports = [5]
            self.alu_ports = [0,1,5,6]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_knl"]:
            self.fp_add_ports = [0,1]
            self.fp_mul_ports = [0,1]
            self.avx512_ports = [0,1]
            self.vec_alu_ports = [0,1]
            self.fp_shuf_ports = [0]
            self.alu_ports = [2,3]
            self.store_port = 5
            self.load_ports = [4,5]
        elif conf["cpu_is_haswell"] or conf["cpu_is_broadwell"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0,1]
            self.vec_alu_ports = [0,1,5]
            self.fp_shuf_ports = [5]
            self.alu_ports = [0,1,5,6]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_sandy"] or conf["cpu_is_ivy"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0]
            self.vec_alu_ports = [0,1,5]
            self.fp_shuf_ports = [5]
            self.alu_ports = [0,1,5]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_westmere"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0]
            self.vec_alu_ports = [0,5]
            self.fp_shuf_ports = [5]
            self.alu_ports = [0,1,5]
            self.store_port = 4
            self.load_ports = [2]
        else:
            raise Exception("Target architecture not modelled.")
        self.spill_ports = self.load_ports
        self.fp_fma_ports = self.fp_mul_ports

        self.num_ports = 9

        self.A = A

        self.insn_names = A.columns.values.tolist()

        insn_indices = {}
        for insn_name in self.insn_names:
            insn_indices[insn_name] = self.insn_names.index(insn_name)
        self.insn_indices = insn_indices

        self.meta_coefs_names = get_meta_coef_names(conf)

        eu_insn_names = [x for x in self.insn_names if x.startswith("eu.")]
        self.num_insns = A[eu_insn_names].sum(axis=1)

    def allocate_cycles_to_ports(self, insn_cycles, port_cycles, ports):
        if len(ports) == 0:
            return

        cycles_remaining = insn_cycles
        neg_filter = insn_cycles < 0
        if sum(neg_filter) > 0:
            ## First stage of reclaim: reduce max down to min across ports:
            f = neg_filter
            ports_min = port_cycles[f][:,ports].min(axis=1)
            for p in ports:
                port_reclaim = np.minimum(port_cycles[f][:,p]-ports_min, -cycles_remaining[f])
                port_cycles[f,p]    -= port_reclaim
                cycles_remaining[f] += port_reclaim

            ## Then spread remaining reclamation equally:
            port_reclaim = -cycles_remaining[f]/float(len(ports))
            for p in ports:
                port_cycles[f,p]    -= port_reclaim
                cycles_remaining[f] += port_reclaim

        pos_filter = np.logical_not(neg_filter)
        if sum(pos_filter) > 0:
            ## First stage of allocation: equalise max and min across ports:
            f = pos_filter
            ports_max = port_cycles[f][:,ports].max(axis=1)
            for p in ports:
                port_alloc = np.minimum(ports_max-port_cycles[f][:,p], cycles_remaining[f])
                port_cycles[f,p]    += port_alloc
                cycles_remaining[f] -= port_alloc

            ## Now spread remaining cycles equally across ports:
            port_alloc = cycles_remaining[f]/float(len(ports))
            for p in ports:
                port_cycles[f,p]    += port_alloc
                cycles_remaining[f] -= port_alloc

    def apply_model(self, x, do_print=False, verify=False, return_bottleneck=False):
        num_datapoints = self.A.shape[0]
        indices = self.insn_indices
        conf = self.conf

        do_track_contributions = do_print or verify or return_bottleneck

        if conf["do_spill_penalty"]:
            meta_coefs = get_meta_coefs(conf, x)
            spill_penalty = max(meta_coefs["spill_penalty"], 0.0)
            # if "mem.spills" in self.insn_indices:
            if "mem.spills" in self.insn_indices and spill_penalty > 0.0:
                num_spills = self.A.values[:,self.insn_indices["mem.spills"]]
                num_spills_per_insn = np.divide(num_spills.astype('float'), self.num_insns.astype('float'))
                cycle_penalty_per_insn = (num_spills_per_insn * spill_penalty).values
            else:
                cycle_penalty_per_insn = np.zeros(num_datapoints)
        elif conf["do_load_penalty"]:
            meta_coefs = get_meta_coefs(conf, x)
            load_penalty = max(meta_coefs["load_penalty"], 0.0)
            if "mem.loads" in self.insn_indices and load_penalty > 0.0:
                num_loads = self.A.values[:,self.insn_indices["mem.loads"]]
                num_loads_per_insn = np.divide(num_loads.astype('float'), self.num_insns.astype('float'))
                cycle_penalty_per_insn = (num_loads_per_insn * load_penalty).values
            else:
                cycle_penalty_per_insn = np.zeros(num_datapoints)
        else:
            cycle_penalty_per_insn = np.zeros(num_datapoints)

        ## Prepare matrix that will hold cycle consumption of each execution port:
        port_cycles = np.zeros( (num_datapoints, self.num_ports) )

        ## Calculate cycle consumption of each instruction category:
        num_insn_coefs = len(x) - len(self.meta_coefs_names)

        x_insn_coefs = x[0:num_insn_coefs]
        # if "mem.spills" in indices and "mem.loads" in indices:
        #     ## Update: treat spill-loads same as memory-loads:
        #     x_insn_coefs = np.append(x_insn_coefs, x_insn_coefs[indices["mem.loads"]])
        ## Update 2: spill CPI now added elsewhere
        ## Update 3: spill CPI only added elsewhere when predicting, not training:
        if "mem.spills" in indices and (num_insn_coefs == (self.A.shape[1]-1)):
            # Add coef for spills:
            if "mem.loads" in indices:
                x_insn_coefs = np.append(x_insn_coefs, x_insn_coefs[indices["mem.loads"]])
            else:
                raise Exception("Need to add coef for mem.spills but coef for mem.loads is missing")

        x_insn_coefs = [x_insn_coefs] * num_datapoints
        y_insn_cycles = self.A.values*x_insn_coefs

        ## Map cycles to ports. Note that with AVX-512 vectorisation on Skylake, 
        ## ports 0 and 1 are only 256-bytes wide so are fused together 
        ## to process SIMD AVX-512 instructions. Note that port 5 is 
        ## 512-bits wide, but on our Skylake Silver SKU it can only process 
        ## Vec Shuffle operations (ie its FMA exec unit is disabled).

        last_cycles = np.zeros(num_datapoints)

        ## AVX512 FP:
        if "eu.avx512" in indices:
            avx512_idx =  indices["eu.avx512"]
            avx512_cycles = y_insn_cycles[:,avx512_idx]
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,avx512_idx])
            avx512_cycles += penalty
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,0] += avx512_cycles
                port_cycles[:,1] += avx512_cycles
            else:
                self.allocate_cycles_to_ports(avx512_cycles, port_cycles, self.avx512_ports)
        if do_track_contributions:
            walltime_cycles_avx512 = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## FP divs:
        if "eu.fp_div" in indices:
            fp_div_idx =  indices["eu.fp_div"]
            fp_div_cycles = y_insn_cycles[:,fp_div_idx] 
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_div_idx])
            fp_div_cycles += penalty
            port_cycles[:,0] += fp_div_cycles
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,1] += y_insn_cycles[:,fp_div_idx]
        if do_track_contributions:
            walltime_cycles_divs = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        if "eu.fp_div_fast" in indices:
            fp_div_fast_idx =  indices["eu.fp_div_fast"]
            fp_div_fast_cycles = y_insn_cycles[:,fp_div_fast_idx]
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_div_fast_idx])
            fp_div_fast_cycles += penalty
            port_cycles[:,0] += fp_div_fast_cycles
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,1] += y_insn_cycles[:,fp_div_fast_idx]
        if do_track_contributions:
            walltime_cycles_divs_fast = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## FP add:
        if "eu.fp_add" in indices:
            fp_add_idx = indices["eu.fp_add"]
            fp_add_cycles = y_insn_cycles[:,fp_add_idx].copy()
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_add_idx])
            fp_add_cycles += penalty
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,0] += fp_add_cycles
                port_cycles[:,1] += fp_add_cycles
            else:
                self.allocate_cycles_to_ports(fp_add_cycles, port_cycles, self.fp_add_ports)
        ## FP mult:
        if "eu.fp_mul" in indices:
            fp_mul_idx = indices["eu.fp_mul"]
            fp_mul_cycles = y_insn_cycles[:,fp_mul_idx].copy()
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_mul_idx])
            fp_mul_cycles += penalty
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,0] += fp_mul_cycles
                port_cycles[:,1] += fp_mul_cycles
            else:
                self.allocate_cycles_to_ports(fp_mul_cycles, port_cycles, self.fp_mul_ports)
        if do_track_contributions:
            walltime_cycles_fp = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## FP FMA:
        if "eu.fp_fma" in indices:
            fp_fma_idx = indices["eu.fp_fma"]
            fp_fma_cycles = y_insn_cycles[:,fp_fma_idx].copy()
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_fma_idx])
            fp_fma_cycles += penalty
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                port_cycles[:,0] += fp_fma_cycles
                port_cycles[:,1] += fp_fma_cycles
            else:
                self.allocate_cycles_to_ports(fp_fma_cycles, port_cycles, self.fp_fma_ports)
        if do_track_contributions:
            walltime_cycles_fma = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## FP shuffle:
        if "eu.fp_shuffle" in indices:
            fp_shuf_idx = indices["eu.fp_shuffle"]
            fp_shuf_cycles = y_insn_cycles[:,fp_shuf_idx].copy()
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,fp_shuf_idx])
            fp_shuf_cycles += penalty
            self.allocate_cycles_to_ports(fp_shuf_cycles, port_cycles, self.fp_shuf_ports)
        if do_track_contributions:
            walltime_cycles_shuffles = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## Vec ALU:
        if "eu.simd_alu" in indices:
            simd_alu_idx = indices["eu.simd_alu"]
            simd_alu_cycles_remaining = y_insn_cycles[:,simd_alu_idx]
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,simd_alu_idx])
            simd_alu_cycles_remaining += penalty
            if conf["cpu_is_skylake"] and conf["avx512_simd_enabled"]:
                simd_alu_cycles_on_p5 = np.minimum(port_cycles[:,0], simd_alu_cycles_remaining)
                port_cycles[:,5]          += simd_alu_cycles_on_p5
                simd_alu_cycles_remaining -= simd_alu_cycles_on_p5
                port_alloc = simd_alu_cycles_remaining/2.0 ## <- leave as /2.0. Remember port fusion.
                port_cycles[:,0] += port_alloc
                port_cycles[:,1] += port_alloc
                port_cycles[:,5] += port_alloc
            else:
                self.allocate_cycles_to_ports(simd_alu_cycles_remaining, port_cycles, self.vec_alu_ports)
        if do_track_contributions:
            walltime_cycles_simd_alu = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## ALU:
        if "eu.alu" in indices:
            alu_idx = indices["eu.alu"]
            alu_cycles_remaining = y_insn_cycles[:,alu_idx]
            penalty = np.multiply(cycle_penalty_per_insn, self.A.values[:,alu_idx])
            alu_cycles_remaining += penalty
            self.allocate_cycles_to_ports(alu_cycles_remaining, port_cycles, self.alu_ports)

        if do_track_contributions:
            walltime_cycles_alu = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        if conf["do_ignore_loads_stores"]:
            if do_track_contributions:
                walltime_cycles_store = np.zeros(num_datapoints)
                walltime_cycles_load  = np.zeros(num_datapoints)
        else:

            if "mem.stores" in indices:
                stores_idx = indices["mem.stores"]
                port_cycles[:,self.store_port] += y_insn_cycles[:,stores_idx]
            if do_track_contributions:
                walltime_cycles_store = np.subtract(port_cycles.max(axis=1), last_cycles)
                last_cycles = port_cycles.max(axis=1)

            if "mem.loads" in indices:
                loads_idx = indices["mem.loads"]
                self.allocate_cycles_to_ports(y_insn_cycles[:,loads_idx], port_cycles, self.load_ports)
            if "mem.spills" in indices:
                spills_idx = indices["mem.spills"]
                self.allocate_cycles_to_ports(y_insn_cycles[:,spills_idx], port_cycles, self.spill_ports)
            if do_track_contributions:
                ## treat spills same as mem.loads:
                walltime_cycles_load = np.subtract(port_cycles.max(axis=1), last_cycles)
                last_cycles = port_cycles.max(axis=1)

        y_model = port_cycles.max(axis=1)

        if do_track_contributions:
            if do_print:
                print("Cycle consumption of each instruction category (cumulative, not parallel):")

            if do_print and verify:
                print("Verifying")

            do_contribute = [True] * num_insn_coefs

            bottleneck = None

            if bool(np.any(walltime_cycles_avx512 != np.zeros(len(walltime_cycles_avx512)))):
                if do_print:
                    print("AVX512 FP | {0}".format(walltime_cycles_avx512))
                if "eu.avx512" in indices:
                    do_contribute[indices["eu.avx512"]] = True
                    bottleneck = "eu.avx512"
            else:
                if "eu.avx512" in indices:
                    do_contribute[indices["eu.avx512"]] = False

            if bool(np.any(walltime_cycles_divs != np.zeros(len(walltime_cycles_divs)))):
                if do_print:
                    print("DIVS      | {0}".format(walltime_cycles_divs))
                # if "eu.fp_div" in indices:
                do_contribute[indices["eu.fp_div"]] = True
                bottleneck = "eu.fp_div"
            else:
                if "eu.fp_div" in indices:
                    do_contribute[indices["eu.fp_div"]] = False

            if bool(np.any(walltime_cycles_divs_fast != np.zeros(len(walltime_cycles_divs_fast)))):
                if do_print:
                    print("DIVS(fast) | {0}".format(walltime_cycles_divs))
                # if "eu.fp_div_fast" in indices:
                do_contribute[indices["eu.fp_div_fast"]] = True
                bottleneck = "eu.fp_div_fast"
            else:
                if "eu.fp_div_fast" in indices:
                    do_contribute[indices["eu.fp_div_fast"]] = False

            if bool(np.any(walltime_cycles_fp != np.zeros(len(walltime_cycles_fp)))):
                if do_print:
                    print("FP        | {0}".format(walltime_cycles_fp))
                if "eu.fp_add" in indices:
                    do_contribute[indices["eu.fp_add"]] = True
                    bottleneck = "eu.fp"
                if "eu.fp_mul" in indices:
                    do_contribute[indices["eu.fp_mul"]] = True
                    bottleneck = "eu.fp"
            else:
                if "eu.fp_add" in indices:
                    do_contribute[indices["eu.fp_add"]] = False
                if "eu.fp_mul" in indices:
                    do_contribute[indices["eu.fp_mul"]] = False

            if bool(np.any(walltime_cycles_fma != np.zeros(len(walltime_cycles_fma)))):
                if do_print:
                    print("FP FMA    | {0}".format(walltime_cycles_fma))
                if "eu.fp_fma" in indices:
                    do_contribute[indices["eu.fp_fma"]] = True
                    bottleneck = "eu.fp"
            else:
                if "eu.fp_fma" in indices:
                    do_contribute[indices["eu.fp_fma"]] = False

            if bool(np.any(walltime_cycles_shuffles != np.zeros(len(walltime_cycles_shuffles)))):
                if do_print:
                    print("FP SHUF   | {0}".format(walltime_cycles_shuffles))
                if "eu.fp_shuffle" in indices:
                    do_contribute[indices["eu.fp_shuffle"]] = True
                    bottleneck = "eu.fp_shuffle"
            else:
                if "eu.fp_shuffle" in indices:
                    do_contribute[indices["eu.fp_shuffle"]] = False

            if bool(np.any(walltime_cycles_simd_alu != np.zeros(len(walltime_cycles_simd_alu)))):
                if do_print:
                    print("SIMD ALU  | {0}".format(walltime_cycles_simd_alu))
                if "eu.simd_alu" in indices:
                    do_contribute[indices["eu.simd_alu"]] = True
                    bottleneck = "eu.simd_alu"
            else:
                if "eu.simd_alu" in indices:
                    do_contribute[indices["eu.simd_alu"]] = False

            if bool(np.any(walltime_cycles_alu != np.zeros(len(walltime_cycles_alu)))):
                if do_print:
                    print("ALU       | {0}".format(walltime_cycles_alu))
                if "eu.alu" in indices:
                    do_contribute[indices["eu.alu"]] = True
                    bottleneck = "eu.alu"
            else:
                if "eu.alu" in indices:
                    do_contribute[indices["eu.alu"]] = False

            if bool(np.any(walltime_cycles_store != np.zeros(len(walltime_cycles_store)))):
                if do_print:
                    print("STORES    | {0}".format(walltime_cycles_store))
                if "mem.stores" in indices:
                    do_contribute[indices["mem.stores"]] = True
                    bottleneck = "mem.stores"
            else:
                if "mem.stores" in indices:
                    do_contribute[indices["mem.stores"]] = False

            if bool(np.any(walltime_cycles_load != np.zeros(len(walltime_cycles_load)))):
                if do_print:
                    # print("LOADS     | {0}".format(walltime_cycles_load))
                    print("LD + SPILL| {0}".format(walltime_cycles_load))
                if "mem.loads" in indices:
                    do_contribute[indices["mem.loads"]] = True
                    bottleneck = "mem.loads"
            else:
                if "mem.loads" in indices:
                    do_contribute[indices["mem.loads"]] = False

            if not bottleneck is None:
                if conf["do_spill_penalty"] and meta_coefs["spill_penalty"] != 0.0:
                    bottleneck += ";spill_penalty={0}".format(meta_coefs["spill_penalty"])
                elif conf["do_load_penalty"] and meta_coefs["load_penalty"] != 0.0:
                    bottleneck += ";load_penalty={0}".format(meta_coefs["load_penalty"])

            if do_print:
                non_contributing_insns = []
                for insn_name in indices:
                    insn_idx = indices[insn_name]
                    if insn_idx >= num_insn_coefs:
                        continue
                    if not do_contribute[insn_idx]:
                        # print("Insn '{0}' does not contribute to overall clock cycle consumption.".format(insn_name))
                        non_contributing_insns.append(insn_name)
                if len(non_contributing_insns) > 0:
                    print("These insns do not contribute to overall clock cycle consumption:")
                    print(non_contributing_insns)

            if verify:
                return (y_model, do_contribute)

            if return_bottleneck:
                return (y_model, bottleneck)

        return y_model

    def predict(self, coefs, do_print=True, return_bottleneck=True):
        coefs_raw = []
        if type(coefs) is dict:
            for n in self.insn_names:
                if n in coefs:
                    coefs_raw.append(coefs[n])
                elif n == "mem.spills" and "mem.loads" in coefs:
                    coefs_raw.append(coefs["mem.loads"])
            for n in self.meta_coefs_names:
                if n in coefs:
                    coefs_raw.append(coefs[n])
        else:
            coefs_raw = coefs

        if return_bottleneck:
            p, bottleneck = self.apply_model(coefs_raw, do_print=do_print, return_bottleneck=True)
            return p, bottleneck
        else:
            p = self.apply_model(coefs_raw, do_print=do_print, return_bottleneck=False)
            return p
