############################################################
############################################################
# Note:
#
# There is no particular reason for this logic to be coded 
# in R. In fact, I now regret choosing R over Python, as 
# R runtime errors do not report the line number.
############################################################

get_script_filepath <- function() {
    cmd_args <- commandArgs(trailingOnly = FALSE)
    needle <- "--file="
    match <- grep(needle, cmd_args)
    if (length(match) > 0) {
        # Rscript
        fp <- normalizePath(sub(needle, "", cmd_args[match]))
    } else {
        # 'source'd via R console
        fp <- normalizePath(sys.frames()[[1]]$ofile)
    }
    return(dirname(fp))
}

library(reshape)

script_dirpath <- get_script_filepath()

source(file.path(script_dirpath, "../utils.R"))

reserved_col_names <- c('Num.threads', 'PAPI.counter', 'Flux.variant', "niters", "kernel")

##########################################################
## Model config:
##########################################################

data_classes <- c('flux.update', 'flux', 'update', 'compute_step', 'time_step', 'up', 'down', 'indirect_rw')

model_conf_params <- c()
model_conf_params <- c(model_conf_params, "do_spill_penalty")
model_conf_params <- c(model_conf_params, "cpu_model")
model_conf_params <- c(model_conf_params, "model_fitting_strategy")
model_conf_params <- c(model_conf_params, "baseline_kernel")
model_conf_params <- c(model_conf_params, "relative_project_direction")

model_fitting_strategy_datums <- list(miniDifferences="miniDifferences", miniAbsolute="miniAbsolute")
# model_fitting_strategy_values <- model_fitting_strategy_datums
model_fitting_strategy_values <- c("miniDifferences")

relative_model_fitting_baselines <- list(Normal="Normal", FluxCripple="FluxCripple")
# baseline_kernel_values <- relative_model_fitting_baselines
# baseline_kernel_values <- c("Normal")
baseline_kernel_values <- c("FluxCripple")

relative_projection_directions <- list(fromMini="fromMini", fromMiniLean="fromMiniLean")
# relative_project_direction_values <- relative_projection_directions
relative_project_direction_values <- c("fromMiniLean")
# relative_project_direction_values <- c("fromMini")

kernels_to_ignore <- c('compute_step', 'time_step', 'up', 'down')
data_cols_to_ignore <- c()
for (l in 0:3) {
    for (k in kernels_to_ignore) {
        data_cols_to_ignore <- c(paste0(k, l), data_cols_to_ignore)
    }
}

preprocess_input_csv <- function(D) {
    if ("Flux.options" %in% names(D)) {
        D$Flux.options <- as.character(D$Flux.options)
        if ("Flux.variant" %in% names(D)) {
            D$Flux.variant <- as.character(D$Flux.variant)

            D <- D[D$Flux.variant == "Normal" | D$Flux.variant == "FluxCripple",]

            # filter <- (D$Flux.variant == "Normal") & (D$Flux.options != "")
            filter <- (D$Flux.options != "")
            D$Flux.variant[filter] <- paste0(D$Flux.variant[filter], "-", D$Flux.options[filter])
        } else {
            D$Flux.variant <- D$Flux.options
        }
        D$Flux.options <- NULL
    }

    ## Drop non-varying cols:
    data_col_names <- get_data_col_names(D)
    for (col in names(D)) {
        if (col %in% reserved_col_names) {
            next
        }
        if (col %in% data_col_names) {
            next
        }
        if (length(unique(D[,col]))==1) {
            D[,col] <- NULL
        }
    }

    if (!("var_id" %in% names(D))) {
        cols_to_concat <- setdiff(names(D), c(reserved_col_names, data_col_names, "Num.threads"))
        D <- concat_cols(D, cols_to_concat, "var_id", TRUE)
    }

    for (dc in data_cols_to_ignore) {
        if (dc %in% names(D)) {
            D[,dc] <- NULL
        }
    }

    return(D)
}

##########################################
## Process instruction counts csv:
##########################################
ic <- read.csv("instruction-counts.mean.csv")

categorise_instructions <- function(ic) {
    insns <- c()
    mem_event_classes <- c()
    for (n in names(ic)) {
        if (startsWith(n, "insn.")) {
            if (n != tolower(n)) {
                ic <- rename_col(ic, n, tolower(n))
                n <- tolower(n)
            }

            if (n == "insn.loads") {
                ic <- rename_col(ic, "insn.loads", "mem.loads")
                mem_event_classes <- c(mem_event_classes, "mem.loads")
            }
            else if (n == "insn.stores") {
                ic <- rename_col(ic, "insn.stores", "mem.stores")
                mem_event_classes <- c(mem_event_classes, "mem.stores")
            } else {
                insns <- c(insns, n)
            }
        }
    }

    exec_unit_instructions <- list()
    exec_unit_instructions[["alu"]] <- c("loop", "cmp", "inc", "j.*", "lea", "add", "mov", "movslq", "movap*", "sar", "[v]?movdq.*", "sh[rl]", "nop", "movzbl", "xor")
    # exec_unit_instructions[["simd_log"]] <- c("vxorpd", "vmovq")
    # exec_unit_instructions[["simd_alu"]] <- c("vpaddd", "vpmulld")
    exec_unit_instructions[["simd_alu"]] <- c("[v]?pand[d]?", "vandps", "[v]?xorp.*", "vpxor[d]?", "vmovq", "vmovdq.*", "[v]?paddd", "[v]?psubd", "[v]?pmulld", "vpinsrd", "[v]?punpck.*", "vextracti128")
    exec_unit_instructions[["simd_shuffle"]] <- c("[v]?unpck.*", "vinsertf128", "vperm.*", "vpgather[dq]d", "vgather[dq]pd", "pshufd", "vpblendmq", "vpmovsxdq", "vbroadcast.*", "[v]?pmovzx.*")
    exec_unit_instructions[["fp_add"]] <- c("[v]?add[sp]d", "[v]?sub[sp]d", "vpsubq", "vmax[sp]d")
    exec_unit_instructions[["fp_mul"]] <- c("mulsd", "[v]?mul[sp]d", "vf[n]?m[as].*")
    exec_unit_instructions[["fp_div"]] <- c("[v]?div[sp]d", "[v]?sqrt[sp]d")
    exec_unit_instructions[["fp_div_fast"]] <- c("vrcp.*", "vrsqrt14pd", "vrsqrt28[sp]d")
    exec_unit_instructions[["fp_mov"]] <- c("[v]?movd", "[v]?movsd", "[v]?movup[sd]", "[v]?movhp[sd]", "[v]?movap[sd]")
    exec_unit_instructions[["avx512_alu"]] <- c("vpxorq", "vptestm.*", "kandw", "kandn.*", "knot.*", "kxorw", "kxnorw")
    exec_unit_instructions[["avx512_shuffle"]] <- c("valign[dq]", "vscatter[dq]p[sd]", "vinserti64x4", "vpbroadcastm.*", "vpbroadcast[bwdq]", "kunpckbw")
    exec_unit_instructions[["avx512_misc"]] <- c("vfpclasspd", "vplzcnt[dq]", "vpconflictd", "vpternlog[dq]", "vfixupimm[sp]d", "kmov[wbqd]", "kshiftrw", "vgetexp[sp][sd]", "vgetmant[sp][sd]", "vscalef[sp][sd]")

    # exec_unit_instructions[["fp_div"]] <- c(exec_unit_instructions[["fp_div"]], exec_unit_instructions[["fp_div_fast"]])
    # exec_unit_instructions[["fp_div_fast"]] <- NULL

    ## Note: I am unsure how to categorise dppd instruction. My data suggest low-cost, but Agner claims 13 cycles.
    exec_unit_instructions[["fp_dppd"]] <- c("[v]?dppd")
    ## ... however, I do not have time to figure this out, so put dppd back into 'fp_add':
    exec_unit_instructions[["fp_add"]] <- c(exec_unit_instructions[["fp_add"]], exec_unit_instructions[["fp_dppd"]])
    exec_unit_instructions[["fp_dppd"]] <- NULL

    exec_units <- names(exec_unit_instructions)

    exec_unit_colnames <- paste0("eu.", exec_units)
    for (euc in exec_unit_colnames) {
        ic[,euc] <- 0
    }

    for (insn in insns) {
        insn_name <- strsplit(insn, ".", fixed=TRUE)[[1]][2]
        exec_unit_found <- FALSE
        for (eu in exec_units) {
            if (insn_name %in% exec_unit_instructions[[eu]]) {
                exec_unit_found <- TRUE
                ic[,paste0("eu.",eu)] <- ic[,paste0("eu.",eu)] + ic[,insn]
                ic[,insn] <- NULL
                break
            }
        }
        if (!exec_unit_found) {
            ## No exact match found. But entries are regex-compatible, so search for matches:
            for (eu in exec_units) {
                for (eu_insn in exec_unit_instructions[[eu]]) {
                    rr <- grep(paste0("^",eu_insn,"$"), insn_name)
                    if (length(rr) > 0) {
                        exec_unit_found <- TRUE
                        ic[,paste0("eu.",eu)] <- ic[,paste0("eu.",eu)] + ic[,insn]
                        ic[,insn] <- NULL
                        break
                    }
                }
                if (exec_unit_found) {
                    break
                }
            }
        }

        if (!exec_unit_found) {
            stop(paste0("Cannot map insn '", insn_name, "' to execution unit."))
        }
    }

    ## Current Intel documentation does not describe how AVX512 instructions are scheduled to 
    ## execution ports, so for now merge with other categories:
    ic["eu.simd_alu"] <- ic["eu.simd_alu"] + ic["eu.avx512_alu"]
    ic["eu.avx512_alu"] <- NULL
    ic["eu.simd_shuffle"] <- ic["eu.simd_shuffle"] + ic["eu.avx512_shuffle"]
    ic["eu.avx512_shuffle"] <- NULL
    ic["eu.fp_mov"] <- ic["eu.fp_mov"] + ic["eu.avx512_misc"]
    ic["eu.avx512_misc"] <- NULL

    return(ic)
}

ic <- categorise_instructions(ic)

exec_unit_colnames <- c()
mem_event_colnames <- c()
for (cn in names(ic)) {
    if (endsWith(cn, ".loads") || endsWith(cn, ".stores")) {
        mem_event_colnames <- c(mem_event_colnames, cn)
    } else if (startsWith(cn, "eu.")) {
        exec_unit_colnames <- c(exec_unit_colnames, cn)
    }
}

if ("Size" %in% names(ic)) {
    ic <- ic[ic$Size==ic$Size[1],]
    ic$Size <- NULL
}

if ("CC.version" %in% names(ic)) {
    ic$CC.version <- NULL
}
ic$kernel <- as.character(ic$kernel)
ic[ic$kernel=="compute_flux_edge", "kernel"] <- "flux"

ic <- preprocess_input_csv(ic)
# write.csv(ic, "instruction-counts.categorised.csv", row.names=FALSE)
#########################################

#################################################################################
## Read in and preprocess csvs:
#################################################################################
if (file.exists("PAPI.mean.csv")) {
    papi_data <- read.csv("PAPI.mean.csv")
} else {
    papi_data <- read.csv("papi.mean.csv")
}
time_data <- read.csv("Times.mean.csv")
if (file.exists("LoopNumIters.mean.csv")) {
    loop_niters_data <- read.csv("LoopNumIters.mean.csv")
} else {
    loop_niters_data <- read.csv("LoopStats.median.csv")
}

cpu <- time_data$CPU[1]

papi_data_names <- names(papi_data)
papi_counter_names <- unique(papi_data$PAPI.counter)

data_col_names <- c()
for (l in seq(0,3)) {
    data_col_names <- c(data_col_names, intersect(paste0(data_classes, l), papi_data_names))
}
data_col_names <- c(data_col_names, intersect(paste0(data_classes, ".total"), papi_data_names))
data_col_names <- c(data_col_names, intersect(c('total','Total'), papi_data_names))

papi_data <- preprocess_input_csv(papi_data)
if (nrow(papi_data)==0) {
    stop("No papi data left after preprocess()")
}
time_data <- preprocess_input_csv(time_data)
if (nrow(time_data)==0) {
    stop("No time data left after preprocess()")
}
loop_niters_data <- preprocess_input_csv(loop_niters_data)
if (nrow(loop_niters_data)==0) {
    stop("No loop_niters_data left after preprocess()")
}

## Discard unnecessary PAPI events:
# necessary_papi_events <- c("PAPI_TOT_CYC", "PAPI_TOT_CYC_MAX", "PAPI_TOT_INS")
necessary_papi_events <- c("PAPI_TOT_CYC", "PAPI_TOT_CYC_MAX", "PAPI_TOT_INS", "PAPI_TOT_INS_MAX")
for (pe in necessary_papi_events) {
    if (!(pe %in% papi_counter_names)) {
        stop(paste0("ERROR: Event '", pe, "' not in papi data csv."))
    }
}
papi_events_to_to_keep <- c(necessary_papi_events)
papi_data <- papi_data[papi_data$PAPI.counter %in% papi_events_to_to_keep,]
papi_data$PAPI.counter <- as.character(papi_data$PAPI.counter)

## Convert the "<kernel><level>" columns into rows:
papi_data <- reshape_data_cols(papi_data, c("flux", "indirect_rw"))
papi_data <- rename_col(papi_data, "value", "count")
time_data <- reshape_data_cols(time_data, c("flux", "indirect_rw"))
time_data <- rename_col(time_data, "value", "runtime")
loop_niters_data <- reshape_data_cols(loop_niters_data, c("flux", "indirect_rw"))
loop_niters_data <- rename_col(loop_niters_data, "value", "niters")

## Transpose the PAPI counters, from rows to columns:
# papi_data <- cast(papi_data, var_id+Flux.variant+level+kernel+Num.threads~PAPI.counter, value="count")
formula <- "var_id"
for (cn in names(papi_data)) {
    if (!(cn %in% c("var_id", "count", "PAPI.counter"))) {
        formula <- paste0(formula, "+", cn)
    }
}
formula <- paste0(formula, "~", "PAPI.counter")
papi_data <- cast(papi_data, as.formula(formula), value="count")

## Adjust 'niters' to account for SIMD width:
loop_niters_data <- split_col(loop_niters_data, "var_id")
if (!("SIMD.len" %in% names(loop_niters_data))) {
    ## Assume that SIMD was disabled.
} else {
    loop_niters_data$SIMD.len <- as.numeric(loop_niters_data$SIMD.len)
    loop_niters_data$niters <- (loop_niters_data$niters + loop_niters_data$SIMD.len - 1) / loop_niters_data$SIMD.len
}
cols_to_concat <- setdiff(names(loop_niters_data), c(reserved_col_names, data_col_names, papi_events_to_to_keep, "niters", "kernel", "level"))
loop_niters_data <- concat_cols(loop_niters_data, cols_to_concat, "var_id", TRUE)

#################################################################################

#################################################################################
## Merge all data into a single table:
#################################################################################
## Merge PAPI and time data:
nrow_pre <- nrow(papi_data)
perf_data <- merge(papi_data, time_data, all=FALSE)
nrow_post <- nrow(perf_data)
if (nrow_pre != nrow_post) {
    stop(paste(nrow_pre, "rows before merge(papi_data, time_data) but", nrow_post, "rows after"))
}

## Merge in loop iteration counts:
nrow_pre <- nrow(perf_data)
perf_data <- merge(perf_data, loop_niters_data, all=FALSE)
nrow_post <- nrow(perf_data)
if (nrow_pre != nrow_post) {
    stop(paste(nrow_pre, "rows before merge(perf_data, loop_niters_data) but", nrow_post, "rows after"))
}

## Merge in instruction counts:
nrow_pre <- nrow(perf_data)
perf_data <- merge(perf_data, ic)
nrow_post <- nrow(perf_data)
if (nrow_pre != nrow_post) {
    print(paste(nrow_pre, "rows before merge(perf_data, ic) but", nrow_post, "rows after"))
    q()
}
#################################################################################

eu_and_mem_colnames <- intersect(names(perf_data), c(exec_unit_colnames, mem_event_colnames))
eu_colnames <- intersect(names(perf_data), c(exec_unit_colnames))

#################################################################################
## Filter data for easier debugging:
#################################################################################
perf_data <- split_col(perf_data, "var_id")
# if ("AVX512" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="AVX512",]
# }
# if ("AVX2" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="AVX2",]
# }
# if ("SSE42" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="SSE42",]
# }
if ("level" %in% names(perf_data)) {
    perf_data <- perf_data[perf_data$level==0,]
    perf_data$level <- NULL
}
if (nrow(perf_data)==0) {
    stop("perf_data is empty")
}

perf_data$CPU <- cpu
write.csv(perf_data, "merged_performance_data.csv", row.names=FALSE)
perf_data$CPU <- NULL
## Now, strip out multi-threaded performance data for the following modelling
if ("KMP.hw.subset" %in% names(perf_data)) {
    perf_data$KMP.hw.subset <- NULL
}
if ("OpenMP" %in% names(perf_data)) {
    perf_data <- perf_data[perf_data$OpenMP=="Off",]
    perf_data[,"OpenMP"] <- NULL
}
if ("Permit.scatter.OpenMP" %in% names(perf_data)) {
    perf_data <- perf_data[perf_data$Permit.scatter.OpenMP=="N",]
    perf_data[,"Permit.scatter.OpenMP"] <- NULL
}
cols_to_concat <- setdiff(names(perf_data), c(reserved_col_names, data_col_names, papi_events_to_to_keep, "runtime", "kernel", eu_and_mem_colnames))
perf_data <- concat_cols(perf_data, cols_to_concat, "var_id", TRUE)

#################################################################################

#################################################################################
## Verify that #assembly * #num_iters == PAPI_TOT_INS:
#################################################################################
perf_data$PAPI_TOT_INS.expected <- 0
for (eu_col in eu_colnames) {
    perf_data$PAPI_TOT_INS.expected <- (perf_data[,eu_col] * perf_data[,"niters"]) + perf_data$PAPI_TOT_INS.expected
}
# perf_data$PAPI_TOT_INS.piter <- round(perf_data$PAPI_TOT_INS / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.piter <- round(perf_data$PAPI_TOT_INS_MAX / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.expected_piter <- round(perf_data$PAPI_TOT_INS.expected / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.eu_diff <- perf_data$PAPI_TOT_INS.piter - perf_data$PAPI_TOT_INS.expected_piter
## My handling of loop unrolling is almost perfect but not quite, so allow 
## a small difference of 1 between expected and actual #instructions/iteration:
bad_run_threshold <- rep(1.0, nrow(perf_data))
## Allow a higher deviation for AVX-512 due to unpredictability of 
## AVX-512-CD masked remainder loops:
bad_run_threshold[grep("AVX512", perf_data$var_id)] <- 66.0
bad_runs <- abs(perf_data$PAPI_TOT_INS.eu_diff) > bad_run_threshold
if (sum(bad_runs)>0) {
    print("Error: Assembly counts do not match with PAPI_TOT_INS for these runs:")
    print(perf_data[bad_runs,c("var_id", "Flux.variant", "PAPI_TOT_INS.piter", "PAPI_TOT_INS.expected_piter", "PAPI_TOT_INS.eu_diff")])
    write.csv(perf_data, "perf_data.csv", row.names=FALSE)
    q()
}
perf_data$PAPI_TOT_INS.expected <- NULL
perf_data$PAPI_TOT_INS.piter <- NULL
perf_data$PAPI_TOT_INS.expected_piter <- NULL
perf_data$PAPI_TOT_INS.eu_diff <- NULL
print("Good, sum of instructions agrees with PAPI_TOT_INS")
#################################################################################

## Clean perf_data
for (col in c("Num.threads")) {
    if (length(unique(perf_data[,col]))==1) {
        perf_data[,col] <- NULL
    }
}
## Update: focus on predicting grind time
perf_data$wg_cycles <- perf_data$PAPI_TOT_CYC_MAX / perf_data$niters
perf_data$wg_sec    <- perf_data$runtime / perf_data$niters

# cpu <- infer_run_value("CPU", raw_source_data_dirpath)
cpu_is_xeon <- length(grep("Xeon",cpu))>0
cpu_is_phi <- cpu_is_xeon && (length(grep("Phi",cpu))>0)
cpu_is_knl <- cpu_is_phi && (length(grep("7210",cpu))>0)
cpu_is_skylake <- cpu_is_xeon && (length(grep("Silver",cpu))>0)
cpu_is_broadwell <- cpu_is_xeon && (length(grep("v4",cpu))>0)
cpu_is_haswell <- length(grep("i5-4",cpu))>0
cpu_is_sandy <- length(grep("i5-2",cpu))>0
cpu_is_westmere <- cpu_is_xeon && (length(grep("X5650",cpu))>0)

perf_data_master <- data.frame(perf_data)
eu_and_mem_colnames_master <- eu_and_mem_colnames
eu_colnames_master <- eu_colnames

## Iterate over all model params:
num_runs <- 1
first_pass <- TRUE
this_run_num = 0
# do_spill_penalty_values <- c(TRUE, FALSE)
# do_spill_penalty_values <- TRUE
do_spill_penalty_values <- FALSE
for (do_spill_penalty in do_spill_penalty_values) {
    if (first_pass) num_runs <- num_runs*length(do_spill_penalty_values)

for (model_fitting_strategy in model_fitting_strategy_values) {
    if (first_pass) num_runs <- num_runs*length(model_fitting_strategy_values)

for (baseline_kernel in baseline_kernel_values) {
    if (first_pass) num_runs <- num_runs*length(baseline_kernel_values)

for (relative_project_direction in relative_project_direction_values) {
    if (first_pass) {
        num_runs <- num_runs*length(relative_project_direction_values)
    }

    if (first_pass) {
        first_pass <- FALSE
    }
    this_run_num <- this_run_num + 1
    print("")
    print("")
    print(paste("Run", this_run_num, "of", num_runs))

    perf_data <- data.frame(perf_data_master)
    eu_and_mem_colnames <- eu_and_mem_colnames_master
    eu_colnames <- eu_colnames_master

#################################################################################

## Write our model params for Python to read in:
model_config_df <- data.frame("key"=character(), "value"=numeric())
model_config_df <- rbind(model_config_df, data.frame("key"="do_spill_penalty", "value"=do_spill_penalty))
model_config_df <- rbind(model_config_df, data.frame("key"="model_fitting_strategy", "value"=model_fitting_strategy))
model_config_df <- rbind(model_config_df, data.frame("key"="baseline_kernel", "value"=baseline_kernel))
model_config_df <- rbind(model_config_df, data.frame("key"="relative_project_direction", "value"=relative_project_direction))
if (cpu_is_skylake)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_skylake", "value"=cpu_is_skylake))
else if (cpu_is_broadwell)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_broadwell", "value"=cpu_is_broadwell))
else if (cpu_is_knl)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_knl", "value"=cpu_is_knl))
else if (cpu_is_westmere)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_westmere", "value"=cpu_is_westmere))
else if (cpu_is_haswell)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_haswell", "value"=cpu_is_haswell))
else if (cpu_is_sandy)
    model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_sandy", "value"=cpu_is_sandy))
else {
    stop(paste0("Do not know how to interpret this CPU: '", cpu, "'"))
}

#################################################################################

#################################################################################
## Preprocess eu data to reduce number of 'features' for better model fitting:
#################################################################################
# perf_data[,"eu.stores"] <- perf_data[,"eu.loads"] + perf_data[,"eu.stores"]
# perf_data[,"eu.loads"] <- NULL

# ## Merge FP_add with FP_mul:
# perf_data[,"eu.fp_mul"] <- perf_data[,"eu.fp_add"] + perf_data[,"eu.fp_mul"]
# perf_data[,"eu.fp_add"] <- NULL

perf_data[,"eu.fp_div"] <- perf_data[,"eu.fp_div_fast"] + perf_data[,"eu.fp_div"]
perf_data[,"eu.fp_div_fast"] <- NULL
## You may think combining 'fp_div' and 'fp_div_fast' is a poor choice. However, I have not 
## seen compiler generate a loop that uses both. It either selects high-precision 
## accurate divs/sqrts, or it selects the approximation versions.

perf_data[,"eu.fp_mov"] <- perf_data[,"eu.fp_mov"] + perf_data[,"eu.simd_shuffle"]
perf_data[,"eu.simd_shuffle"] <- NULL

# # ## Remove instruction categories that should have no impact:
# perf_data[,"eu.alu"] <- NULL
# perf_data[,"eu.simd_alu"] <- NULL
# perf_data[,"eu.stores"] <- NULL
# perf_data[,"eu.fp_mov"] <- NULL
# perf_data[,"eu.simd_shuffle"] <- NULL
# perf_data[,"eu.loads"] <- NULL

eu_and_mem_colnames <- intersect(names(perf_data), c(exec_unit_colnames, mem_event_colnames))
eu_colnames <- intersect(names(perf_data), c(exec_unit_colnames))
#################################################################################

#################################################################################
## Construct linear system:
#################################################################################
model_data_flux <- perf_data[perf_data$kernel != "indirect_rw",]
model_data_flux$kernel <- NULL

lin_systems <- data.frame(model_data_flux)
# data_col_names <- setdiff(names(lin_systems), c(mandatory_columns, "var_id", "Flux.variant", "kernel", "level", "niters"))
data_col_names <- setdiff(names(lin_systems), c("var_id", "Flux.variant", "kernel", "level", "niters"))
## First linear system is now ready (absolute performance data)

## Construct second linear system consisting of performance differences between MG-CFD variants:
lin_systems_relative <- data.frame(lin_systems)
lin_systems_relative[,data_col_names] <- 0
for (v in unique(lin_systems$var_id)) {
    v_filter <- lin_systems$var_id==v

    if (baseline_kernel == relative_model_fitting_baselines[["FluxCripple"]]) {
        # print("Using FluxCripple as baseline")
        baseline_filter <- v_filter & (lin_systems$Flux.variant=="FluxCripple")
        baseline <- lin_systems[baseline_filter,]
    } else {
        # print("Using 'Normal' as baseline (almost most expensive variant)")
        ## Use variant with almost most instructions as the baseline:
        ls_v_ordered <- data.frame(lin_systems[v_filter,])
        ls_v_ordered <- ls_v_ordered[order(-ls_v_ordered$wg_cycles),]
        ls_v_ordered <- ls_v_ordered[grepl("Normal", ls_v_ordered$Flux.variant),]
        ls_v_ordered <- ls_v_ordered[ls_v_ordered$Flux.variant != "Normal-PrecomputeLength;",]
        mini_baseline_variant <- ls_v_ordered$Flux.variant[1]
        baseline <- ls_v_ordered[ls_v_ordered$Flux.variant==mini_baseline_variant,]
    }
    if (nrow(baseline) != 1) {
        print(paste("nrow(baseline) =", nrow(baseline)))
        stop(paste("Failed to get baseline for var_id =", v))
    }

    for (f in data_col_names) {
        if (baseline_kernel == relative_model_fitting_baselines[["FluxCripple"]]) {
            lin_systems_relative[v_filter, f] <- lin_systems[v_filter, f] - baseline[1,f]
        } else {
            lin_systems_relative[v_filter, f] <- baseline[1,f] - lin_systems[v_filter, f]
        }
    }
}

## Filter-out baseline (drop.sum = 0):
lin_systems_relative[,"drop.sum"] <- 0
for (eu_col in eu_and_mem_colnames) {
    lin_systems_relative[,"drop.sum"] <- abs(lin_systems_relative[,eu_col]) + lin_systems_relative[,"drop.sum"]
}
lin_systems_relative <- lin_systems_relative[lin_systems_relative$drop.sum != 0,]
lin_systems_relative$drop.sum <- NULL

#################################################################################
## Begin model fitting and prediction:
#################################################################################

cpi_estimates <- NULL
# mini_wg_cycles <- NULL
model_coefs <- c()

# proj_row_template <- data.frame(do_spill_penalty = do_spill_penalty, 
#                                 model_fitting_strategy=model_fitting_strategy, 
#                                 baseline_kernel=baseline_kernel, 
#                                 relative_project_direction=relative_project_direction, 
#                                 model_error_pct = 100.0, 
#                                 mini_cycles = 0.0,
#                                 fc = 0.0)
proj_row_template <- data.frame(do_spill_penalty = do_spill_penalty, 
                                model_fitting_strategy=model_fitting_strategy, 
                                baseline_kernel=baseline_kernel, 
                                relative_project_direction=relative_project_direction)

var_vals <- unique(lin_systems_relative$var_id)
for (var_id in var_vals) {
    ## Iterate over each 'var' and fit model:

    proj_row <- data.frame(proj_row_template)
    proj_row$var_id <- as.character(var_id)

    print(paste("Fitting model to var_id =", var_id))
    if (length(model_fitting_strategy_values) > 1) {
        print(paste("  model_fitting_strategy =", proj_row_template$model_fitting_strategy))
    }
    if (length(baseline_kernel_values) > 1) {
        print(paste("  baseline_kernel =", proj_row_template$baseline_kernel))
    }
    if (length(relative_project_direction_values) > 1) {
        print(paste("  relative_project_direction =", proj_row_template$relative_project_direction))
    }

    ## Select the linear system for model fitting:
    if (model_fitting_strategy == model_fitting_strategy_datums[["miniDifferences"]]) {
        # print("  Using relative linear system")
        var_lin_system <- lin_systems_relative[lin_systems_relative$var_id==var_id,]
    } else {
        # print("  Using absolute linear system")
        var_lin_system <- lin_systems[lin_systems$var_id==var_id,]
    }

    ## Need to know whether vectorisation is enabled, as depending on 
    ## the architecture it can cause execution ports to fuse:
    model_conf <- data.frame(model_config_df)
    var_lin_system_var_split <- split_col(var_lin_system, "var_id")
    if (!("SIMD.len" %in% names(var_lin_system_var_split))) {
        ## Assume that SIMD was disabled.
        simd_len = 1
    } else {
        simd_len = var_lin_system_var_split$SIMD.len[1]
    }
    isa = var_lin_system_var_split$Instruction.set[1]
    if (simd_len > 1) {
        # if (isa == "AVX512" && simd_len <= 8) {
        if (isa == "AVX512") {
            model_conf <- rbind(model_conf, data.frame("key"="avx512_simd_enabled", "value"=TRUE))
        }
    }
    if (model_fitting_strategy == model_fitting_strategy_datums[["miniDifferences"]]) {
        model_conf <- rbind(model_conf, data.frame("key"="predict_perf_diff", "value"=TRUE))
    }
    model_conf <- model_conf[model_conf$key!="baseline_kernel",]
    model_conf <- model_conf[model_conf$key!="model_fitting_strategy",]
    model_conf <- model_conf[model_conf$key!="relative_project_direction",]

    ## Write out model fitting data for Python:
    model_coef_colnames <- c()
    for (eu_name in eu_and_mem_colnames) {
        if (length(unique(var_lin_system[,eu_name])) > 1) {
            model_coef_colnames <- c(model_coef_colnames, eu_name)
        }
    }
    model_data <- var_lin_system[,c("Flux.variant", "wg_cycles", model_coef_colnames)]
    # model_fitting_data <- model_data[model_data$Flux.variant!="iflux",]
    # model_fitting_data <- model_fitting_data[model_fitting_data$Flux.variant!="FluxCripple",]
    model_fitting_data <- model_data[model_data$Flux.variant!="FluxCripple",]
    ## Remove duplicate rows:
    model_fitting_data <- model_fitting_data[!duplicated(model_fitting_data[,model_coef_colnames]),]
    ## Round:
    model_fitting_data[,"wg_cycles"] <- round(model_fitting_data[,"wg_cycles"], digits=1)
    if (!dir.exists("Modelling")) {
        dir.create("Modelling")
    } else {
        ## Empty dir
        do.call(file.remove, list(list.files("Modelling/", full.names = TRUE)))
    }
    write.csv(model_fitting_data, file.path("Modelling", "fitting_data.csv"), row.names=FALSE)
    write.csv(model_conf, file.path("Modelling", "insn_model_conf.csv"), row.names=FALSE)

    ## Run Python solver to estimate CPIs:
    python_output <- system(paste("python", file.path(script_dirpath, "model_interface.py"), "-f"), intern=TRUE)
    solution_filepath <- file.path("Modelling", "solution.csv")
    if (!file.exists(solution_filepath)) {
        print(paste0("ERROR: Python script did not complete and write solution to csv: ", solution_filepath))
        print(python_output)
        q()
    }
    # print("-----------------------------")
    # print("Python solver output:")
    # print("--------------")
    # for (line in python_output) {
    #     print(line)
    # }
    # print("-----------------------------")

    ## Read in model coefficients:
    run_model_coefs <- read.csv(solution_filepath)
    for (i in 1:nrow(run_model_coefs)) {
        coef <- as.character(run_model_coefs$coef[i])
        cpi  <- run_model_coefs$cpi[i]
        proj_row[coef] <- cpi

        if (!is.null(cpi_estimates)) {
            if (!(coef %in% names(cpi_estimates))) {
                if (nrow(cpi_estimates) == 0) {
                    cpi_estimates[,coef] <- numeric()
                } else {
                    cpi_estimates[,coef] <- 0.0
                }
            }
        }
    }
    model_coefs <- union(model_coefs, as.character(run_model_coefs$coef))
    for (coef in setdiff(model_coefs, names(proj_row))) {
        proj_row[,coef] <- 0.0
    }

    var_perf_data <- perf_data[perf_data$var_id==var_id,]
    var_perf_data$var_id <- NULL
    mini_cycles <- var_perf_data[(var_perf_data$Flux.variant=="Normal")&(var_perf_data$kernel=="flux"),"PAPI_TOT_CYC"]
    mini_niters <- var_perf_data[(var_perf_data$Flux.variant=="Normal")&(var_perf_data$kernel=="flux"),"niters"]
    rw_cycles   <- var_perf_data[(var_perf_data$Flux.variant=="Normal")&(var_perf_data$kernel=="indirect_rw"),"PAPI_TOT_CYC"]
    #######################################################
    ## Append this modelling to main data frame:
    if (is.null(cpi_estimates)) {
        cpi_estimates <- proj_row
    } else {
        cpi_estimates <- safe_rbind(cpi_estimates, proj_row)
    }
}

}
}
}
}

print("Finished fitting, now writing out.")

## Remove unused model params:
for (p in model_conf_params) {
    if (p %in% names(cpi_estimates)) {
        if (length(unique(cpi_estimates[,p]))==1) {
            cpi_estimates[,p] <- NULL
        }
    }
}
model_conf_params <- intersect(model_conf_params, names(cpi_estimates))
if (length(model_conf_params) > 0) {
    cpi_estimates <- cpi_estimates[do.call(order, cpi_estimates[,c("var_id", model_conf_params)]),]
} else {
    cpi_estimates <- cpi_estimates[order(cpi_estimates$var_id),]
}

## Round numbers:
for (coef in model_coefs) {
    cpi_estimates[,coef] <- round(cpi_estimates[,coef], digits=3)
}
for (col in c("mini_cycles", "target_cycles", "target_cycles_model")) {
    if (col %in% names(cpi_estimates)) {
        cpi_estimates[,col] <- round(cpi_estimates[,col])
    }
}
for (col in c("model_error_pct", "fc")) {
    if (col %in% names(cpi_estimates)) {
        cpi_estimates[,col] <- round(cpi_estimates[,col], digits=3)
    }
}

if ("model_params" %in% names(cpi_estimates)) {
    cpi_estimates[,"model_params"] <- NULL
}

projections_filename <- "cpi_estimates.csv"
print(paste("CPI estimates written to", projections_filename))
write.csv(cpi_estimates, projections_filename, row.names=FALSE)

# mini_wg_cycles_filename <- "wg_cycles.csv"
# print(paste("Source wg cycles written to", mini_wg_cycles_filename))
# write.csv(mini_wg_cycles, mini_wg_cycles_filename, row.names=FALSE)

## Cleanup:
system(paste("rm -r", "Modelling"))
