import numpy as np
import os, sys

script_dirpath = os.path.dirname(os.path.realpath(__file__))

import imp
imp.load_source('Utils', os.path.join(script_dirpath, "../Utils.py"))
from Utils import *

class ArchModel(object):
    def __init__(self, conf, A):
        self.conf = conf

        if conf["cpu_is_skylake"]:
            self.fp_add_ports = [0,1]
            self.fp_mul_ports = [0,1]
            self.vec_alu_ports = [0,1,5]
            self.fp_mov_ports = [5]
            self.alu_ports = [0,1,5,6]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_knl"]:
            self.fp_add_ports = [0,1]
            self.fp_mul_ports = [0,1]
            self.vec_alu_ports = [0,1]
            self.fp_mov_ports = 0
            self.alu_ports = [2,3]
            self.store_port = [5]
            self.load_ports = [4,5]
        elif conf["cpu_is_haswell"] or conf["cpu_is_broadwell"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0,1]
            self.vec_alu_ports = [0,1,5]
            self.fp_mov_ports = [5]
            self.alu_ports = [0,1,5,6]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_sandy"] or conf["cpu_is_ivy"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0]
            self.vec_alu_ports = [0,2,5]
            self.fp_mov_ports = [0,5]
            self.alu_ports = [0,1,5]
            self.store_port = 4
            self.load_ports = [2,3]
        elif conf["cpu_is_westmere"]:
            self.fp_add_ports = [1]
            self.fp_mul_ports = [0]
            self.vec_alu_ports = [0,5]
            self.fp_mov_ports = [0,5]
            self.alu_ports = [0,1,5]
            self.store_port = 4
            self.load_ports = [2]
        else:
            raise Exception("Target architecture not modelled.")
        self.num_ports = 9

        self.A = A

        self.insn_names = A.columns.values.tolist()

        insn_indices = {}
        for insn_name in self.insn_names:
            insn_indices[insn_name] = self.insn_names.index(insn_name)
        self.insn_indices = insn_indices

        self.meta_coefs_names = get_meta_coef_names(conf)

    def allocate_cycles_to_ports(self, insn_cycles, port_cycles, ports):
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

    def apply_model(self, x, do_print=False, verify=False):

        ## Prepare matrix that will hold cycle consumption of each execution port:
        num_datapoints = self.A.shape[0]
        port_cycles = np.zeros( (num_datapoints, self.num_ports))

        ## Calculate cycle consumption of each instruction category:
        num_insn_coefs = len(x) - len(self.meta_coefs_names)
        x_insn_coefs = [x[0:num_insn_coefs]] * num_datapoints
        y_insn_cycles = self.A.values*x_insn_coefs

        ## Map cycles to ports. Note that with AVX-512 vectorisation on Skylake, 
        ## ports 0 and 1 are only 256-bytes wide so are fused together 
        ## to process SIMD AVX-512 instructions. Note that port 5 is 
        ## 512-bits wide, but on our Skylake Silver SKU it can only process 
        ## Vec Shuffle operations (ie its FMA exec unit is disabled).

        ## FP divs:
        if "eu.fp_div" in self.insn_indices:
            fp_div_idx =  self.insn_indices["eu.fp_div"]
            port_cycles[:,0] += y_insn_cycles[:,fp_div_idx]
            if self.conf["cpu_is_skylake"] and self.conf["avx512_simd_enabled"]:
                port_cycles[:,1] += y_insn_cycles[:,fp_div_idx]
        if "eu.fp_div_fast" in self.insn_indices:
            port_cycles[:,0] += y_insn_cycles[:,fp_div_fast_idx]
            if self.conf["cpu_is_skylake"] and self.conf["avx512_simd_enabled"]:
                port_cycles[:,1] += y_insn_cycles[:,fp_div_fast_idx]

        if do_print or verify:
            walltime_cycles_divs = port_cycles.max(axis=1)
            last_cycles = port_cycles.max(axis=1)

        ## FP add:
        if "eu.fp_add" in self.insn_indices:
            fp_add_idx = self.insn_indices["eu.fp_add"]
            fp_add_cycles = y_insn_cycles[:,fp_add_idx]
            if self.conf["cpu_is_skylake"] and self.conf["avx512_simd_enabled"]:
                port_cycles[:,0] += fp_add_cycles
                port_cycles[:,1] += fp_add_cycles
            else:
                self.allocate_cycles_to_ports(fp_add_cycles, port_cycles, self.fp_add_ports)
        ## FP mult:
        if "eu.fp_mul" in self.insn_indices:
            fp_mul_idx = self.insn_indices["eu.fp_mul"]
            fp_mul_cycles = y_insn_cycles[:,fp_mul_idx]
            if self.conf["cpu_is_skylake"] and self.conf["avx512_simd_enabled"]:
                port_cycles[:,0] += fp_mul_cycles
                port_cycles[:,1] += fp_mul_cycles
            else:
                self.allocate_cycles_to_ports(fp_mul_cycles, port_cycles, self.fp_mul_ports)
        if do_print or verify:
            walltime_cycles_fp = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## mov:
        if "eu.fp_mov" in self.insn_indices:
            fp_mov_idx = self.insn_indices["eu.fp_mov"]
            # port_cycles[:,self.fp_mov_ports] += y_insn_cycles[:,fp_mov_idx]
            self.allocate_cycles_to_ports(y_insn_cycles[:,fp_mov_idx], port_cycles, self.fp_mov_ports)
        ## simd shuf:
        if "eu.simd_shuffle" in self.insn_indices:
            simd_shuffle_idx = self.insn_indices["eu.simd_shuffle"]
            port_cycles[:,self.fp_mov_ports] += y_insn_cycles[:,simd_shuffle_idx]
        if do_print or verify:
            walltime_cycles_movs = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## Vec ALU:
        if "eu.simd_alu" in self.insn_indices:
            simd_alu_idx = self.insn_indices["eu.simd_alu"]
            simd_alu_cycles_remaining = y_insn_cycles[:,simd_alu_idx]
            if self.conf["cpu_is_skylake"] and self.conf["avx512_simd_enabled"]:
                simd_alu_cycles_on_p5 = np.minimum(port_cycles[:,0], simd_alu_cycles_remaining)
                port_cycles[:,5]          += simd_alu_cycles_on_p5
                simd_alu_cycles_remaining -= simd_alu_cycles_on_p5
                port_alloc = simd_alu_cycles_remaining/2.0 ## <- leave as /2.0. Remember port fusion.
                port_cycles[:,0] += port_alloc
                port_cycles[:,1] += port_alloc
                port_cycles[:,5] += port_alloc
            else:
                self.allocate_cycles_to_ports(simd_alu_cycles_remaining, port_cycles, self.vec_alu_ports)
        if do_print or verify:
            walltime_cycles_simd_alu = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        ## ALU:
        if "eu.alu" in self.insn_indices:
            alu_idx = self.insn_indices["eu.alu"]
            alu_cycles_remaining = y_insn_cycles[:,alu_idx]
            self.allocate_cycles_to_ports(alu_cycles_remaining, port_cycles, self.alu_ports)

        if do_print or verify:
            walltime_cycles_alu = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        if "mem.stores" in self.insn_indices:
            stores_idx = self.insn_indices["mem.stores"]
            port_cycles[:,self.store_port] += y_insn_cycles[:,stores_idx]
        if do_print or verify:
            walltime_cycles_store = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        if "mem.loads" in self.insn_indices:
            loads_idx = self.insn_indices["mem.loads"]
            self.allocate_cycles_to_ports(y_insn_cycles[:,loads_idx], port_cycles, self.load_ports)
        if do_print or verify:
            walltime_cycles_load = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        y_model = port_cycles.max(axis=1)

        if self.conf["do_spill_penalty"]:
            meta_coefs = get_meta_coefs(self.conf, x)
            # y_model += meta_coefs["spill_penalty"] * self.A.values[:,stores_idx]
            y_model += meta_coefs["spill_penalty"] * self.A.values[:,loads_idx]
        if do_print or verify:
            walltime_cycles_spills = np.subtract(port_cycles.max(axis=1), last_cycles)
            last_cycles = port_cycles.max(axis=1)

        if do_print or verify:
            if do_print:
                print("Cycle consumption of each instruction category (cumulative, not parallel):")

            if do_print and verify:
                print("Verifying")

            do_contribute = [True] * len(x)

            if bool(np.any(walltime_cycles_divs != np.zeros(len(walltime_cycles_divs)))):
                if do_print:
                    print("DIVS      | {0}".format(walltime_cycles_divs))
                do_contribute[self.insn_indices["eu.fp_div"]] = True
            else:
                do_contribute[self.insn_indices["eu.fp_div"]] = False

            if bool(np.any(walltime_cycles_fp != np.zeros(len(walltime_cycles_fp)))):
                if do_print:
                    print("FP        | {0}".format(walltime_cycles_fp))
                if "eu.fp_add" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.fp_add"]] = True
                if "eu.fp_mul" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.fp_mul"]] = True
            else:
                if "eu.fp_add" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.fp_add"]] = False
                if "eu.fp_mul" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.fp_mul"]] = False

            if bool(np.any(walltime_cycles_movs != np.zeros(len(walltime_cycles_movs)))):
                if do_print:
                    print("MOV/SHUF  | {0}".format(walltime_cycles_movs))
                do_contribute[self.insn_indices["eu.fp_mov"]] = True
            else:
                do_contribute[self.insn_indices["eu.fp_mov"]] = False

            if bool(np.any(walltime_cycles_simd_alu != np.zeros(len(walltime_cycles_simd_alu)))):
                if do_print:
                    print("SIMD ALU  | {0}".format(walltime_cycles_simd_alu))
                do_contribute[self.insn_indices["eu.simd_alu"]] = True
            else:
                if "eu.simd_alu" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.simd_alu"]] = False

            if bool(np.any(walltime_cycles_alu != np.zeros(len(walltime_cycles_alu)))):
                if do_print:
                    print("ALU       | {0}".format(walltime_cycles_alu))
                if "eu.alu" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.alu"]] = True
            else:
                if "eu.alu" in self.insn_indices:
                    do_contribute[self.insn_indices["eu.alu"]] = False

            if bool(np.any(walltime_cycles_store != np.zeros(len(walltime_cycles_store)))):
                if do_print:
                    print("STORES    | {0}".format(walltime_cycles_store))
                if "mem.stores" in self.insn_indices:
                    do_contribute[self.insn_indices["mem.stores"]] = True
            else:
                if "mem.stores" in self.insn_indices:
                    do_contribute[self.insn_indices["mem.stores"]] = False

            if bool(np.any(walltime_cycles_load != np.zeros(len(walltime_cycles_load)))):
                if do_print:
                    print("LOADS     | {0}".format(walltime_cycles_load))
                if "mem.loads" in self.insn_indices:
                    do_contribute[self.insn_indices["mem.loads"]] = True
            else:
                if "mem.loads" in self.insn_indices:
                    do_contribute[self.insn_indices["mem.loads"]] = False

            if bool(np.any(walltime_cycles_spills != np.zeros(len(walltime_cycles_spills)))):
                if do_print:
                    print("SPILL PEN.| {0}".format(walltime_cycles_mem))

            if do_print:
                non_contributing_insns = []
                for insn_name in self.insn_indices:
                    if not do_contribute[self.insn_indices[insn_name]]:
                        # print("Insn '{0}' does not contribute to overall clock cycle consumption.".format(insn_name))
                        non_contributing_insns.append(insn_name)
                if len(non_contributing_insns) > 0:
                    print("These insns do not contribute to overall clock cycle consumption:")
                    print(non_contributing_insns)

            if verify:
                return (y_model, do_contribute)

        return y_model

    def predict(self, coefs):
        coefs_raw = []
        if type(coefs) is dict:
            for n in self.insn_names:
                coefs_raw.append(coefs[n])
            for n in self.meta_coefs_names:
                coefs_raw.append(coefs[n])
        else:
            coefs_raw = coefs

        p = self.apply_model(coefs_raw, do_print=True)
        # p = self.apply_model(coefs_raw)
        return p
