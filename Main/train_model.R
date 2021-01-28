library(parallel)
library(MASS)
numCores <- detectCores(logical=FALSE)
numCores <- numCores-1
library(flock)

############################################################
############################################################
# Note:
#
# There is no particular reason for this logic to be coded 
# in R. In fact, I now regret choosing R over Python, as 
# R runtime errors do not report the line number.
############################################################

# debug_model_fitting <- TRUE
debug_model_fitting <- FALSE

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

# reserved_col_names <- c('Num.threads', 'PAPI.counter', 'Flux.variant', "niters", "kernel", "CC")
## Until I get precise FP working on KNL with AVX-512, I must keep compilers separate. 
## Reason is that if both divs and divs-fast are in model fitting data, the model fails 
## to get to CPI of div-fast and I do not know why.
## By having "CC" in reserved_col_names, it ends up being ignored by model fitting.
reserved_col_names <- c('Num.threads', 'PAPI.counter', 'Flux.variant', "niters", "kernel")

mandatory_columns <- c("Instruction.set", "CPU", "CC")

##########################################################
## Model config:
##########################################################

data_classes <- c('flux.update', 'flux', 'update', 'compute_step', 'time_step', 'restrict', 'prolong', 'unstructured_stream', 'compute_stream', 'unstructured_compute')

model_conf_params <- c()
model_conf_params <- c(model_conf_params, "do_spill_penalty")
model_conf_params <- c(model_conf_params, "do_load_penalty")
model_conf_params <- c(model_conf_params, "do_prune_insn_classes")
model_conf_params <- c(model_conf_params, "do_ignore_loads_stores")
model_conf_params <- c(model_conf_params, "cpu_model")
model_conf_params <- c(model_conf_params, "model_fitting_strategy")
model_conf_params <- c(model_conf_params, "baseline_kernel")
# model_conf_params <- c(model_conf_params, "relative_project_direction")

# do_spill_penalty_values <- c(FALSE, TRUE)
do_spill_penalty_values <- c(FALSE)
# do_spill_penalty_values <- c(TRUE)

# do_load_penalty_values <- c(FALSE, TRUE)
do_load_penalty_values <- c(FALSE)
# do_load_penalty_values <- c(TRUE)

# do_prune_insn_classes_values <- c(FALSE, TRUE)
# do_prune_insn_classes_values <- c(FALSE)
do_prune_insn_classes_values <- c(TRUE)

# do_ignore_loads_stores_values <- c(FALSE, TRUE)
# do_ignore_loads_stores_values <- c(TRUE)
do_ignore_loads_stores_values <- c(FALSE)

model_fitting_strategy_datums <- list(miniDifferences="miniDifferences", miniAbsolute="miniAbsolute")
model_fitting_strategy_values <- unlist(model_fitting_strategy_datums, use.names=FALSE)
# model_fitting_strategy_values <- c("miniDifferences")
# model_fitting_strategy_values <- c("miniAbsolute")

relative_model_fitting_baselines <- list(Normal="Normal", FluxSynthetic="Flux-Synthetic")
# baseline_kernel_values <- unlist(relative_model_fitting_baselines, use.names=FALSE)
# baseline_kernel_values <- c("Normal")
baseline_kernel_values <- c("Flux-Synthetic")

# relative_projection_directions <- list(fromMini="fromMini", fromMiniLean="fromMiniLean")
# relative_project_direction_values <- unlist(relative_projection_directions, use.names=FALSE)
# relative_project_direction_values <- c("fromMiniLean")
# # relative_project_direction_values <- c("fromMini")

optimisation_search_algorithms <- list(basin="basin", shgo="shgo")
# optimisation_search_algorithm_values <- unlist(optimisation_search_algorithms, use.names=FALSE)
optimisation_search_algorithm_values <- c("basin")
# optimisation_search_algorithm_values <- c("shgo")

## 100 iterations should be enough to complete local minimisation, but 
## set to 150 to be sure.
# basin_local_iters_values <- c(150)
# ## Do not need many jumps:
# basin_jump_values <- c(10)
# basin_step_values <- c(1)
## UPDATE: fitting model to SIMD performance data needs more searching, particulary to 
##         find best CPI estimate for eu.simd_fp_div (should be ~38 for Intel-AVX/AVX2, ~34 for Clang)
basin_local_iters_values <- c(400)
basin_jump_values <- c(15)
basin_step_values <- c(2)

# basin_local_iters_values <- c(250, 300)
# basin_jump_values <- c(15, 50)
# basin_step_values <- c(1, 2)

# basin_local_iters_values <- c(100, 150, 250)
# basin_jump_values <- c(10)
# basin_step_values <- c(1, 2, 3)

# Fast basin for debugging:
# basin_local_iters_values <- c(100)
# basin_jump_values <- c(50)
# basin_step_values <- c(1)

kernels_to_ignore <- c('compute_step', 'time_step', 'restrict', 'prolong')
data_cols_to_ignore <- c()
for (l in 0:3) {
    for (k in kernels_to_ignore) {
        data_cols_to_ignore <- c(paste0(k, l), data_cols_to_ignore)
    }
}

isa_group <- "Intel"

preprocess_input_csv <- function(D) {
    if ("Flux.options" %in% names(D)) {
        D$Flux.options <- as.character(D$Flux.options)
        # if ("Flux.variant" %in% names(D)) {
        #     D$Flux.variant <- as.character(D$Flux.variant)

        #     D <- D[D$Flux.variant == "Normal" | D$Flux.variant == "Flux-Synthetic",]

        #     filter <- D$Flux.options != ""
        #     D$Flux.variant[filter] <- paste0(D$Flux.variant[filter], "-", D$Flux.options[filter])
        # } else {
        #     D$Flux.variant <- D$Flux.options
        # }
        f <- D$Flux.options != ""
        D$Flux.variant[f] <- paste0("Flux-", D$Flux.options[f])
        D$Flux.variant[!f] <- "Normal"
        D$Flux.options <- NULL
    }

    if ("CC.version" %in% names(D)) {
        D$CC.version <- NULL
    }
    if ("MG.cycles" %in% names(D)) {
        D$MG.cycles <- NULL
    }

    for (l in c(0,1,2,3)) {
        c1 <- paste0("compute_flux_edge", l)
        if (c1 %in% names(D)) {
            D <- rename_col(D, c1, paste0("flux",l))
        }
    }

    if ("ISA.group" %in% names(D)) {
        if (length(unique(D$ISA.group)) > 1) {
            stop("Training data must only have one ISA group (Intel/ARM)")
        }
        isa_group <- D$ISA.group[1]
        D$ISA.group <- NULL
    }

    if ("SIMD.failed" %in% names(D)) {
        f <- D$SIMD.failed == "True"
        if ("SIMD.len" %in% names(D)) {
            D$SIMD.len[f] <- 1
        }
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
        if (col %in% mandatory_columns) {
            next
        }
        if (length(unique(D[,col]))==1) {
            D[,col] <- NULL
        }
    }

    # if ("Num.threads" %in% names(D)) {
    #     D <- D[D$Num.threads==1,]
    # }
    # if ("Mesh" %in% names(D)) {
    #     ## Keep the one mesh with most data:
    #     meshes <- unique(D$Mesh)
    #     mesh_counts <- c()
    #     for (m in meshes) {
    #         mesh_counts[m] <- sum(D$Mesh==m)
    #     }
    #     largest_m <- D$Mesh[1]
    #     largest_m_count <- mesh_counts[largest_m]
    #     for (m in meshes) {
    #         mc <- mesh_counts[m]
    #         if (mc > largest_m_count) {
    #             largest_m <- m
    #             largest_m_count <- mc
    #         }
    #     }
    #     D <- D[D$Mesh==largest_m,]
    #     D[,"Mesh"] <- NULL
    # }

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
ic <- NULL
ic_filename_candidates <- c("instruction-counts.mean.csv", "instruction-counts.csv")
for (icf in ic_filename_candidates) {
    if (file.exists(icf)) {
        ic <- read.csv(icf)
        break
    }
}
if (is.null(ic)) {
    stop("instruction counts csv not present")
}

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
            } else if (n == "insn.load_spills") {
                ic <- rename_col(ic, "insn.load_spills", "mem.load_spills")
                mem_event_classes <- c(mem_event_classes, "mem.load_spills")
            }
            else if (n == "insn.store_spills") {
                ic <- rename_col(ic, "insn.store_spills", "mem.store_spills")
                mem_event_classes <- c(mem_event_classes, "mem.store_spills")
            } else {
                insns <- c(insns, n)
            }
        }
    }

    exec_unit_instructions <- list()
    if (isa_group == "ARM") {
        exec_unit_mapping_filepath <- file.path(script_dirpath, "Backend", "ARM-instructions.csv")
    } else {
        exec_unit_mapping_filepath <- file.path(script_dirpath, "Backend", "Intel-instructions.csv")
    }
    exec_unit_mapping <- read.csv(exec_unit_mapping_filepath)
    exec_unit_mapping[,"instruction"] <- as.character(exec_unit_mapping[,"instruction"])
    exec_unit_mapping[,"exec_unit"] <- as.character(exec_unit_mapping[,"exec_unit"])
    for (eu in unique(exec_unit_mapping[,"exec_unit"])) {
        exec_unit_instructions[[eu]] <- c()
    }
    for (i in 1:nrow(exec_unit_mapping)) {
        insn <- exec_unit_mapping[i,"instruction"]
        eu <- exec_unit_mapping[i,"exec_unit"]
        if (eu %in% names(exec_unit_instructions)) {
            exec_unit_instructions[[eu]] <- c(exec_unit_instructions[[eu]], insn)
        } else {
            exec_unit_instructions[[eu]] <- c(insn)
        }
    }
    ## Dot product instructions use two exec units: fp_mul and fp_shuffle, but my model requires a 
    ## mapping to one so use fp_mul:
    exec_unit_instructions[["fp_mul"]] <- c(exec_unit_instructions[["fp_mul"]], exec_unit_instructions[["fp_dpp"]])
    exec_unit_instructions[["fp_dpp"]] <- NULL

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
    ic[,"eu.DISCARD"] <- NULL

    return(ic)
}

ic <- categorise_instructions(ic)

exec_unit_colnames <- c()
mem_event_colnames <- c()
for (cn in names(ic)) {
    if (startsWith(cn, "mem.")) {
        mem_event_colnames <- c(mem_event_colnames, cn)
    } else if (startsWith(cn, "eu.")) {
        exec_unit_colnames <- c(exec_unit_colnames, cn)
    }
}

if ("Size" %in% names(ic)) {
    ic <- ic[ic$Size==ic$Size[1],]
    ic$Size <- NULL
}

ic$kernel <- as.character(ic$kernel)
ic[ic$kernel=="compute_flux_edge", "kernel"] <- "flux"

## 'Flux options' did not affect unstructured_compute performance, so prune data:
ic$Flux.options <- as.character(ic$Flux.options)
f <- ic$kernel=="unstructured_compute" & ic$Flux.options==""
ic_syn <- ic[f,]
ic_syn$kernel <- "flux"
ic_syn$Flux.options <- "Synthetic"
ic <- safe_rbind(ic[ic$kernel!="unstructured_compute",], ic_syn)

## If no spills were detected, infer from difference between flux and unstructured_stream kernels. 
## From analysis of Intel assembly, most of this difference are spills.
## "Most" => roughly 75% of extra loads, and 100% of extra stores, are for spills.
if ("unstructured_stream" %in% names(ic)) {
    if (!("mem.load_spills" %in% names(ic))) {
        ic[,"mem.load_spills"] <- 0
    }
    if (!("mem.store_spills" %in% names(ic))) {
        ic[,"mem.store_spills"] <- 0
    }
    ic_flux <- ic[ic$kernel=="flux",]
    ic_rw   <- ic[ic$kernel=="unstructured_stream",]
    ## Duplicate rw data for unstructured_compute kernel:
    ic_rw_uc <- ic_rw[ic_rw$Flux.options == "",]
    ic_rw_uc$Flux.options <- "Synthetic"
    ic_rw <- safe_rbind(ic_rw, ic_rw_uc)
    for (cn in exec_unit_colnames) {
        ic_rw[,cn] <- NULL
    }
    for (cn in mem_event_colnames) {
        ic_rw <- rename_col(ic_rw, cn, paste0(cn, ".rw"))
    }
    ic_rw$kernel <- NULL
    ic_flux <- merge(ic_flux, ic_rw)
    f <- ic_flux[,"mem.load_spills"]==0
    ic_flux[f,"mem.load_spills"] = (ic_flux[f,"mem.loads"] - ic_flux[f,"mem.loads.rw"]) * 0.75
    ic_flux[f,"mem.loads"] = ic_flux[f,"mem.loads"] - ic_flux[f,"mem.load_spills"]
    f <- ic_flux[,"mem.store_spills"]==0
    ic_flux[f,"mem.store_spills"] = ic_flux[f,"mem.stores"] - ic_flux[f,"mem.stores.rw"]
    ic_flux[f,"mem.stores"] = ic_flux[f,"mem.stores"] - ic_flux[f,"mem.store_spills"]
    ic_flux[,"mem.loads.rw"] <- NULL
    ic_flux[,"mem.stores.rw"] <- NULL
    ic_flux[,"mem.load_spills.rw"] <- NULL
    ic_flux[,"mem.store_spills.rw"] <- NULL
    ic_rw <- ic[ic$kernel=="unstructured_stream",]
    ic <- safe_rbind(ic_flux, ic_rw)
}

## Treat spill stores identically to memory stores:
ic[,"mem.stores"] <- ic[,"mem.stores"] +  ic[,"mem.store_spills"]
ic[,"mem.store_spills"] <- NULL
ic <- rename_col(ic, "mem.load_spills", "mem.spills")
mem_event_colnames <- c()
for (cn in names(ic)) {
    if (startsWith(cn, "mem.")) {
        mem_event_colnames <- c(mem_event_colnames, cn)
    }
}

ic <- preprocess_input_csv(ic)
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
loop_niters_data <- NULL
filename_candidates <- c("LoopNumIters.mean.csv", "LoopNumIters.csv", "LoopStats.median.csv", "LoopStats.csv")
for (f in filename_candidates) {
    if (file.exists(f)) {
        loop_niters_data <- read.csv(f)
        break
    }
}
if (is.null(loop_niters_data)) {
    stop("LoopNumIters csv not present")
}
if ("counter" %in% names(loop_niters_data)) {
    loop_niters_data <- loop_niters_data[loop_niters_data["counter"]=="#iterations_MAX",]
    loop_niters_data["counter"] <- NULL
}

if (!("CPU" %in% names(time_data))) {
    stop(paste0("'CPU' column not in input data"))
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
necessary_papi_events <- c("PAPI_TOT_CYC.THREADS_MEAN", "PAPI_TOT_INS.THREADS_MEAN")
for (pe in necessary_papi_events) {
    if (!(pe %in% papi_counter_names)) {
        stop(paste0("ERROR: Event '", pe, "' not in papi data csv."))
    }
}
papi_events_to_to_keep <- c(necessary_papi_events)
papi_data <- papi_data[papi_data$PAPI.counter %in% papi_events_to_to_keep,]
papi_data$PAPI.counter <- as.character(papi_data$PAPI.counter)

## Convert the "<kernel><level>" columns into rows:
papi_data <- reshape_data_cols(papi_data, c("flux", "unstructured_stream"))
papi_data <- rename_col(papi_data, "value", "count")
time_data <- reshape_data_cols(time_data, c("flux", "unstructured_stream"))
time_data <- rename_col(time_data, "value", "runtime")
loop_niters_data <- reshape_data_cols(loop_niters_data, c("flux", "unstructured_stream"))
loop_niters_data <- rename_col(loop_niters_data, "value", "niters")

## Transpose the PAPI counters, from rows to columns:
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
perf_data <- perf_data[perf_data$runtime != 0.0,]

## Merge in loop iteration counts:
nrow_pre <- nrow(perf_data)
merge_cols <- intersect(names(perf_data), names(loop_niters_data))
perf_data2 <- merge(perf_data, loop_niters_data, all=FALSE)
nrow_post <- nrow(perf_data2)
if (nrow_pre != nrow_post) {
    print(paste(nrow_pre, "rows before merge(perf_data, loop_niters_data) but", nrow_post, "rows after"))
    print("Merged on these columns:")
    print(merge_cols)
    if (length(merge_cols) == 1) {
        perf_data <- perf_data[order(perf_data[,merge_cols[1]]), c(merge_cols, setdiff(names(perf_data), merge_cols))]
        loop_niters_data <- loop_niters_data[order(loop_niters_data[,merge_cols[1]]), c(merge_cols, setdiff(names(loop_niters_data), merge_cols))]
    } else {
        perf_data <- perf_data[do.call(order, perf_data[,merge_cols]), c(merge_cols, setdiff(names(perf_data), merge_cols))]
        loop_niters_data <- loop_niters_data[do.call(order, loop_niters_data[,merge_cols]), c(merge_cols, setdiff(names(loop_niters_data), merge_cols))]
    }
    write.csv(perf_data, "debug-perf_data.csv", row.names=FALSE)
    write.csv(loop_niters_data, "debug-loop_niters_data.csv", row.names=FALSE)
    q()
}
perf_data <- perf_data2

## Merge in instruction counts:
nrow_pre <- nrow(perf_data)
merge_cols <- intersect(names(perf_data), names(ic))
perf_data <- merge(perf_data, ic)
nrow_post <- nrow(perf_data)
if (nrow_pre != nrow_post) {
    print(paste(nrow_pre, "rows before merge(perf_data, ic) but", nrow_post, "rows after"))
    print("Merged on these columns:")
    print(merge_cols)
    q()
}

write.csv(perf_data, "merged_performance_data.csv", row.names=FALSE)
if ("Num.threads" %in% names(perf_data)) {
    perf_data <- perf_data[perf_data$Num.threads==1,]
}
if ("Mesh" %in% names(perf_data)) {
    ## Keep the one mesh with most data:
    meshes <- unique(perf_data$Mesh)
    mesh_counts <- c()
    for (m in meshes) {
        mesh_counts[m] <- sum(perf_data$Mesh==m)
    }
    largest_m <- perf_data$Mesh[1]
    largest_m_count <- mesh_counts[largest_m]
    for (m in meshes) {
        mc <- mesh_counts[m]
        if (mc > largest_m_count) {
            largest_m <- m
            largest_m_count <- mc
        }
    }
    perf_data <- perf_data[perf_data$Mesh==largest_m,]
    perf_data[,"Mesh"] <- NULL
}

#################################################################################

eu_and_mem_colnames <- intersect(names(perf_data), c(exec_unit_colnames, mem_event_colnames))
eu_colnames <- intersect(names(perf_data), c(exec_unit_colnames))

#################################################################################
## Filter data for easier debugging:
#################################################################################
perf_data <- split_col(perf_data, "var_id")
# if ("Host" %in% perf_data$Instruction.set) {
#    perf_data <- perf_data[perf_data$Instruction.set=="Host",]
# }
# if ("AVX512" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="AVX512",]
# }
# if ("AVX2" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="AVX2",]
# }
# if ("SSE42" %in% perf_data$Instruction.set) {
#     perf_data <- perf_data[perf_data$Instruction.set=="SSE42",]
# }
# if ("level" %in% names(perf_data)) {
#     perf_data <- perf_data[perf_data$level==0,]
# }
# if ("OpenMP" %in% names(perf_data)) {
#     perf_data <- perf_data[perf_data$OpenMP=="Off",]
# }
# if ("CC" %in% names(perf_data)) {
#     perf_data <- perf_data[perf_data$CC=="intel",]
# }
if (nrow(perf_data)==0) {
    stop("perf_data is empty")
}

perf_data$CPU <- cpu
cols_to_keep <- setdiff(names(perf_data), "eu.load")
# perf_data$CPU <- NULL
## Now, strip out multi-threaded performance data for the following modelling
if ("KMP.hw.subset" %in% names(perf_data)) {
    perf_data$KMP.hw.subset <- NULL
}
if ("Permit.scatter.OpenMP" %in% names(perf_data)) {
    perf_data[,"Permit.scatter.OpenMP"] <- NULL
}
cols_to_concat <- setdiff(names(perf_data), c(reserved_col_names, data_col_names, papi_events_to_to_keep, "runtime", "kernel", eu_and_mem_colnames))
cols_to_concat <- setdiff(cols_to_concat, c("CPU"))
perf_data <- concat_cols(perf_data, cols_to_concat, "var_id", TRUE)
# write.csv(perf_data, "merged_performance_data.csv", row.names=FALSE)
perf_data$CPU <- NULL

#################################################################################

#################################################################################
## Verify that #assembly * #num_iters == PAPI_TOT_INS:
#################################################################################
perf_data <- split_col(perf_data, "var_id")
perf_data$PAPI_TOT_INS.expected <- 0
for (eu_col in eu_colnames) {
    perf_data$PAPI_TOT_INS.expected <- (perf_data[,eu_col] * perf_data[,"niters"]) + perf_data$PAPI_TOT_INS.expected
}
# perf_data$PAPI_TOT_INS.piter <- round(perf_data$PAPI_TOT_INS_MAX / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.piter <- round(perf_data$PAPI_TOT_INS.THREADS_MEAN / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.expected_piter <- round(perf_data$PAPI_TOT_INS.expected / perf_data$niters, digits=1)
perf_data$PAPI_TOT_INS.eu_diff <- perf_data$PAPI_TOT_INS.piter - perf_data$PAPI_TOT_INS.expected_piter
## My handling of loop unrolling is almost perfect but not quite, so allow 
## a small difference of 1 between expected and actual #instructions/iteration:
bad_run_threshold <- rep(1.0, nrow(perf_data))
## Allow a higher deviation for AVX-512 due to unpredictability of 
## AVX-512-CD masked remainder loops:
bad_run_threshold[grep("AVX512", perf_data$var_id)] <- 66.0
## Allow a tolerance where manual CA is used, due to nested loop admin instructions:
manual_ca_mask <- grep("Manual", perf_data$SIMD.conflict.avoidance.strategy)
bad_run_threshold[manual_ca_mask] <- 0.1*perf_data[manual_ca_mask, "PAPI_TOT_INS.piter"]

bad_runs <- abs(perf_data$PAPI_TOT_INS.eu_diff) > bad_run_threshold
if (sum(bad_runs)>0) {
    bad_perf_data <- perf_data[bad_runs,]
    if (nrow(bad_perf_data) > 10) {
        bad_perf_data <- bad_perf_data[1:10,]
    }
    print("Error: Assembly counts do not match with PAPI_TOT_INS for these runs:")
    print(bad_perf_data[,c("var_id", "Flux.variant", "PAPI_TOT_INS.piter", "PAPI_TOT_INS.expected_piter", "PAPI_TOT_INS.eu_diff")])
    write.csv(perf_data, "perf_data.csv", row.names=FALSE)
    q()
}
perf_data$PAPI_TOT_INS.expected <- NULL
perf_data$PAPI_TOT_INS.piter <- NULL
perf_data$PAPI_TOT_INS.expected_piter <- NULL
perf_data$PAPI_TOT_INS.eu_diff <- NULL
print("Good, sum of instructions agrees with PAPI_TOT_INS")

cols_to_concat <- setdiff(names(perf_data), c(reserved_col_names, data_col_names, papi_events_to_to_keep, "runtime", "kernel", eu_and_mem_colnames))
cols_to_concat <- setdiff(cols_to_concat, c("CPU"))
perf_data <- concat_cols(perf_data, cols_to_concat, "var_id", TRUE)
#################################################################################

## Clean perf_data
for (col in c("Num.threads")) {
    if (col %in% names(perf_data)) {
        if (length(unique(perf_data[,col]))==1) {
            perf_data[,col] <- NULL
        }
    }
}
## Update: focus on predicting grind time
# perf_data$wg_cycles <- perf_data$PAPI_TOT_CYC.THREADS_MAX / perf_data$niters
perf_data$wg_cycles <- perf_data$PAPI_TOT_CYC.THREADS_MEAN / perf_data$niters
perf_data$wg_sec    <- perf_data$runtime / perf_data$niters

# cpu <- infer_run_value("CPU", raw_source_data_dirpath)
cpu_is_xeon <- length(grep("Xeon",cpu))>0
cpu_is_phi <- cpu_is_xeon && (length(grep("Phi",cpu))>0)
cpu_is_knl <- cpu_is_phi && (length(grep("7210",cpu))>0)
cpu_is_cascade <- cpu_is_xeon && (length(grep("[0-9]2[0-9][0-9]",cpu))>0)
cpu_is_skylake <- cpu_is_xeon && (length(grep("[0-9]1[0-9][0-9]",cpu))>0)
cpu_is_broadwell <- cpu_is_xeon && (length(grep("v4",cpu))>0)
cpu_is_haswell <- length(grep("i5-4",cpu))>0
cpu_is_ivy <- cpu_is_xeon && (length(grep("v2",cpu))>0)
cpu_is_sandy <- length(grep("i5-2",cpu))>0
cpu_is_westmere <- cpu_is_xeon && (length(grep("X5650",cpu))>0)
if (cpu_is_cascade) {
    ## Cascade Lake architecture is same as Skylake
    cpu_is_skylake <- TRUE
}

perf_data_master <- data.frame(perf_data)
eu_and_mem_colnames_master <- eu_and_mem_colnames
eu_colnames_master <- eu_colnames

perf_data <- data.frame(perf_data_master)
eu_and_mem_colnames <- eu_and_mem_colnames_master
eu_colnames <- eu_colnames_master

#################################################################################

#################################################################################
## Preprocess eu data to reduce number of 'features' for better model fitting:
#################################################################################
if ("eu.load" %in% names(perf_data)) {
    ## This category only exists to ensure all instruction are accounted for. 
    ## I will have already counted memory loads separately, so do not need 
    ## this category.
    perf_data[,"eu.load"] <- NULL
}

eu_and_mem_colnames <- intersect(names(perf_data), c(exec_unit_colnames, mem_event_colnames))
eu_colnames <- intersect(names(perf_data), c(exec_unit_colnames))
#################################################################################

## Construct combinations of var_id's and model permutations:
var_vals <- unique(perf_data$var_id)
model_perms <- merge(var_vals, do_spill_penalty_values, all=TRUE)
model_perms <- rename_col(model_perms, "x", "var_id")
model_perms <- rename_col(model_perms, "y", "do_spill_penalty")
model_perms <- merge(model_perms, do_load_penalty_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "do_load_penalty")
model_perms <- merge(model_perms, do_prune_insn_classes_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "do_prune_insn_classes")
model_perms <- merge(model_perms, do_ignore_loads_stores_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "do_ignore_loads_stores")
model_perms <- merge(model_perms, baseline_kernel_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "baseline_kernel")
model_perms <- merge(model_perms, model_fitting_strategy_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "model_fitting_strategy")
model_perms <- merge(model_perms, basin_local_iters_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "opt_num_iters")
model_perms <- merge(model_perms, basin_jump_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "opt_num_jumps")
model_perms <- merge(model_perms, basin_step_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "opt_step_size")
model_perms <- merge(model_perms, optimisation_search_algorithm_values, all=TRUE)
model_perms <- rename_col(model_perms, "y", "optimisation_search_algorithm")

cols_to_default_to_zero <- c("load_penalty", "spill_penalty")
cols_to_default_to_one <- c("eu.alu", "eu.simd_alu", "eu.fp_shuffle", "eu.fp_fma", "eu.fp_add", "eu.fp_mul", "eu.fp_div", "eu.fp_div_fast", "eu.avx512", "eu.simd_fp_mul", "eu.simd_fp_div", "eu.simd_fp_fast", "mem.loads", "mem.stores", "mem.spills")
lock_filename <- tempfile()
append_and_write_row <- function(r) {
    if (!is.null(r)) {
        # Reload 'cpi_estimates' from filesystem, only way I know of 
        # to synchronise between worker processes:
        projections_filename <- "cpi_estimates.csv"
        if (file.exists(projections_filename)) {
            cpi_estimates <- read.csv(projections_filename, stringsAsFactors=FALSE)
            if (!("optimisation_search_option" %in% names(cpi_estimates))) {
                cpi_estimates <- concat_cols(cpi_estimates, c("iters", "jumps", "steps"), "optimisation_search_option", TRUE)
            }
            cpi_estimates[is.na(cpi_estimates[,"optimisation_search_option"]), "optimisation_search_option"] <- ""
        } else {
            cpi_estimates <- NULL
        }

        if (is.null(cpi_estimates)) {
            cpi_estimates <- r
        } else {
            missing_cols_from_lhs <- setdiff(names(r), names(cpi_estimates))
            for (cn in missing_cols_from_lhs) {
                if (cn %in% cols_to_default_to_zero) {
                    cpi_estimates[,cn] <- 0.0
                }
                else if (cn %in% cols_to_default_to_one) {
                    cpi_estimates[,cn] <- 1.0
                }
            }
            missing_cols_from_rhs <- setdiff(names(cpi_estimates), names(r))
            for (cn in missing_cols_from_rhs) {
                if (cn %in% cols_to_default_to_zero) {
                    r[,cn] <- 0.0
                }
                else if (cn %in% cols_to_default_to_one) {
                    r[,cn] <- 1.0
                }
            }
            cpi_estimates <- safe_rbind(cpi_estimates, r)
        }

        write.csv(cpi_estimates, projections_filename, row.names=FALSE)
    }

    return(cpi_estimates)
}

cpi_estimate_already_exists <- function(var_id, model_perm) {
    projections_filename <- "cpi_estimates.csv"
    if (file.exists(projections_filename)) {
        cpi_estimates <- read.csv(projections_filename, stringsAsFactors=FALSE)
        if (!("optimisation_search_option" %in% names(cpi_estimates))) {
            cpi_estimates <- concat_cols(cpi_estimates, c("iters", "jumps", "steps"), "optimisation_search_option", TRUE)
        }
        cpi_estimates[is.na(cpi_estimates[,"optimisation_search_option"]), "optimisation_search_option"] <- ""
    } else {
        # cpi_estimates <- NULL
        return(FALSE)
    }

    if (nrow(model_perm) != 1) {
        stop(paste("Expected one row in model_perm, not", nrow(model_perm)))
    }

    f <- cpi_estimates$var_id == var_id
    for (mp in names(model_perm)) {
        if (mp %in% names(cpi_estimates)) {
            # print(paste0("Checking mp=",mp))
            rhs <- (cpi_estimates[,mp] == model_perm[1,mp])
            # print("f:")
            # print(f)
            # print("rhs:")
            # print(rhs)
            f <- f & rhs
        }
    }
    # print("Final check")
    if (sum(f) > 0) {
        return(TRUE)
    } else {
        return(FALSE)
    }
}

predict_fn <- function(model_perm_id) {
    # Decode model_perm_id:
    model_perm <- model_perms[model_perm_id,]
    var_id <- as.character(model_perm$var_id)

    baseline_kernel <- as.character(model_perm$baseline_kernel)
    model_fitting_strategy <- as.character(model_perm$model_fitting_strategy)
    do_spill_penalty <- model_perm$do_spill_penalty
    do_load_penalty <- model_perm$do_load_penalty
    do_prune_insn_classes <- model_perm$do_prune_insn_classes
    do_ignore_loads_stores <- model_perm$do_ignore_loads_stores
    optimisation_search_algorithm <- model_perm$optimisation_search_algorithm
    opt_num_iters <- model_perm$opt_num_iters
    opt_num_jumps <- model_perm$opt_num_jumps
    opt_step_size <- model_perm$opt_step_size

    if (optimisation_search_algorithm=="basin") {
        model_perm$optimisation_search_option <- paste0("iters=",opt_num_iters,"^jumps=",opt_num_jumps,"^steps=",opt_step_size)
        model_perm$optimisation_search_cost   <- opt_num_iters*opt_num_jumps
    } else {
        model_perm$optimisation_search_option <- ""
        model_perm$optimisation_search_cost   <- 1
    }

    print(paste("Processing", model_perm_id, "of", nrow(model_perms), "(var_id =", var_id, ")"))

    model_data <- perf_data[perf_data$var_id==var_id,]

    if (do_spill_penalty && do_load_penalty) {
        # Nonsense
        return()
    }

    if (do_load_penalty) {
        ## Spills and loads will be treated 100% identically by model, so just combine here
        model_data[,"mem.loads"] <- model_data[,"mem.loads"] + model_data[,"mem.spills"]
        model_data[,"mem.spills"] <- NULL
    }
    model_eu_and_mem_colnames <- intersect(names(model_data), c(exec_unit_colnames, mem_event_colnames))
    model_eu_colnames <- intersect(names(model_data), c(exec_unit_colnames))

    if (do_prune_insn_classes) {
        merge_mappings <- list()
        merge_mappings[[length(merge_mappings)+1]] <- c("eu.fp_shuffle", "")
        merge_mappings[[length(merge_mappings)+1]] <- c("eu.simd_alu", "eu.alu")
        merge_mappings[[length(merge_mappings)+1]] <- c("eu.fp_fma", "eu.fp_mul")
        merge_mappings[[length(merge_mappings)+1]] <- c("eu.fp_add", "eu.fp_mul")

        merge_mappings[[length(merge_mappings)+1]] <- c("eu.simd_fp_shuffle", "") ## <- newly added
        merge_mappings[[length(merge_mappings)+1]] <- c("eu.simd_fp_add", "eu.simd_fp_mul") ## <- newly added
        for (m in merge_mappings) {
            if (m[1] %in% names(model_data)) {
                if (m[2] == "") {
                    model_data[,m[1]] <- NULL
                }
                else if (m[2] %in% names(model_data)) {
                    model_data[,m[2]] <- model_data[,m[2]] + model_data[,m[1]]
                    model_data[,m[1]] <- NULL
                }
            }
        }
        # Merge AVX-512 FP into FP_MUL IFF not vectorised. Reason: SIMD AVX-512 
        # fuses ports 0&1 so must be handled separately
        if ("eu.avx512" %in% names(model_data)) {
            if ("SIMD" %in% names(model_data)) {
                f <- model_data["SIMD"] == "N"
            } else {
                f <- rep(TRUE, nrow(model_data))
            }
            model_data[f,"eu.fp_mul"] <- model_data[f,"eu.fp_mul"] + model_data[f,"eu.avx512"]
            model_data[f,"eu.avx512"] <- 0
        }

        model_eu_and_mem_colnames <- intersect(names(model_data), c(exec_unit_colnames, mem_event_colnames))
        model_eu_colnames <- intersect(names(model_data), c(exec_unit_colnames))
    }

    if (do_ignore_loads_stores) {
        for (col in c("mem.loads", "mem.stores")) {
            if (col %in% names(model_data)) {
                model_data[,col] <- 0
            }
        }
        model_eu_and_mem_colnames <- intersect(names(model_data), c(exec_unit_colnames, mem_event_colnames))
        model_eu_colnames <- intersect(names(model_data), c(exec_unit_colnames))

        # # Drop indirect-rw kernel:
        # model_data <- model_data[model_data["kernel"]!="unstructured_stream",]
    }

    #################################################################################
    ## Construct linear system:
    #################################################################################
    model_data_flux <- model_data[model_data$kernel != "unstructured_stream",]
    model_data_flux$kernel <- NULL

    lin_systems <- data.frame(model_data_flux)
    data_col_names <- setdiff(names(lin_systems), c("var_id", "Flux.variant", "kernel", "level", "niters", "CC"))
    ## First linear system is now ready (absolute performance data)

    # ## Construct second linear system consisting of performance differences between MG-CFD variants:
    # lin_systems_relative <- data.frame(model_data_flux)
    # lin_systems_relative[,data_col_names] <- 0
    # for (v in unique(lin_systems$var_id)) {
    #     v_filter <- lin_systems$var_id==v

    #     if (baseline_kernel == relative_model_fitting_baselines[["Flux-Synthetic"]]) {
    #         baseline_filter <- v_filter & (lin_systems$Flux.variant=="Flux-Synthetic")
    #         baseline <- lin_systems[baseline_filter,]
    #     } else {
    #         ## Use variant with almost most instructions as the baseline:
    #         ls_v_ordered <- data.frame(lin_systems[v_filter,])
    #         ls_v_ordered <- ls_v_ordered[order(-ls_v_ordered$wg_cycles),]
    #         ls_v_ordered <- ls_v_ordered[grepl("Normal", ls_v_ordered$Flux.variant),]
    #         ls_v_ordered <- ls_v_ordered[ls_v_ordered$Flux.variant != "Flux-PrecomputeLength;",]
    #         mini_baseline_variant <- ls_v_ordered$Flux.variant[1]
    #         baseline <- ls_v_ordered[ls_v_ordered$Flux.variant==mini_baseline_variant,]
    #     }
    #     if (nrow(baseline) != 1) {
    #         # Possible when using multiple compilers.
    #         baseline <- baseline[order(baseline$runtime),]
    #         baseline <- baseline[1,]
    #     }
    #     if (nrow(baseline) != 1) {
    #         print(paste("nrow(baseline) =", nrow(baseline)))
    #         print(baseline)
    #         stop(paste("Failed to get baseline for var_id =", v))
    #     }

    #     for (f in data_col_names) {
    #         if (baseline_kernel == relative_model_fitting_baselines[["Flux-Synthetic"]]) {
    #             lin_systems_relative[v_filter, f] <- lin_systems[v_filter, f] - baseline[1,f]
    #         } else {
    #             lin_systems_relative[v_filter, f] <- baseline[1,f] - lin_systems[v_filter, f]
    #         }
    #     }
    # }

    ## Can fit to both 'flux' and 'unstructured_stream' kernels
    lin_systems <- data.frame(model_data)
    lin_systems$tmp_order <- lin_systems$Flux.variant=="Normal"
    lin_systems <- lin_systems[rev(order(lin_systems$tmp_order)),]
    lin_systems$tmp_order <- NULL
    lin_systems <- lin_systems[order(lin_systems[,"var_id"], lin_systems[,"kernel"]),]

    # ## Filter-out baseline (drop.sum = 0):
    # lin_systems_relative[,"drop.sum"] <- 0
    # for (eu_col in model_eu_and_mem_colnames) {
    #     lin_systems_relative[,"drop.sum"] <- abs(lin_systems_relative[,eu_col]) + lin_systems_relative[,"drop.sum"]
    # }
    # lin_systems_relative <- lin_systems_relative[lin_systems_relative$drop.sum != 0,]
    # lin_systems_relative$drop.sum <- NULL

    #################################################################################

    # model_config_df <- data.frame("key"=character(), "value"=numeric())
    model_config_df <- data.frame("key"=character(), "value"=character())
    model_config_df <- rbind(model_config_df, data.frame("key"="do_spill_penalty", "value"=do_spill_penalty))
    model_config_df <- rbind(model_config_df, data.frame("key"="do_load_penalty", "value"=do_load_penalty))
    model_config_df <- rbind(model_config_df, data.frame("key"="do_prune_insn_classes", "value"=do_prune_insn_classes))
    model_config_df <- rbind(model_config_df, data.frame("key"="do_ignore_loads_stores", "value"=do_ignore_loads_stores))
    model_config_df <- rbind(model_config_df, data.frame("key"="optimisation_search_algorithm", "value"=optimisation_search_algorithm))
    model_config_df <- rbind(model_config_df, data.frame("key"="baseline_kernel", "value"=baseline_kernel))
    model_config_df <- rbind(model_config_df, data.frame("key"="model_fitting_strategy", "value"=model_fitting_strategy))
    if (optimisation_search_algorithm == "basin") {
        model_config_df <- rbind(model_config_df, data.frame("key"="basin_local_iters", "value"=as.character(opt_num_iters)))
        model_config_df <- rbind(model_config_df, data.frame("key"="basin_jumps", "value"=as.character(opt_num_jumps)))
        model_config_df <- rbind(model_config_df, data.frame("key"="basin_steps", "value"=as.character(opt_step_size)))
    }
    if (cpu_is_skylake)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_skylake", "value"="TRUE"))
    else if (cpu_is_broadwell)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_broadwell", "value"="TRUE"))
    else if (cpu_is_knl)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_knl", "value"="TRUE"))
    else if (cpu_is_westmere)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_westmere", "value"="TRUE"))
    else if (cpu_is_haswell)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_haswell", "value"="TRUE"))
    else if (cpu_is_ivy)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_ivy", "value"="TRUE"))
    else if (cpu_is_sandy)
        model_config_df <- rbind(model_config_df, data.frame("key"="cpu_is_sandy", "value"="TRUE"))
    else {
        stop(paste0("Do not know how to interpret this CPU: '", cpu, "'"))
    }

    #################################################################################
    ## Begin model fitting and prediction:
    #################################################################################

    proj_row_template <- data.frame(do_spill_penalty = do_spill_penalty, 
                                    do_load_penalty = do_load_penalty, 
                                    do_prune_insn_classes = do_prune_insn_classes, 
                                    do_ignore_loads_stores = do_ignore_loads_stores, 
                                    model_fitting_strategy=model_fitting_strategy, 
                                    optimisation_search_algorithm=optimisation_search_algorithm,
                                    optimisation_search_option="",
                                    baseline_kernel=baseline_kernel, 
                                    stringsAsFactors=FALSE)
    if (optimisation_search_algorithm=="basin") {
        proj_row_template$optimisation_search_option <- paste0("iters=",opt_num_iters,"^jumps=",opt_num_jumps,"^steps=",opt_step_size)
        proj_row_template$optimisation_search_cost   <- opt_num_iters*opt_num_jumps
    } else {
        proj_row_template$optimisation_search_option <- ""
        proj_row_template$optimisation_search_cost   <- 1
    }

    ## Iterate over each 'var' and fit model:
    proj_row <- data.frame(proj_row_template)
    var_id <- as.character(var_id)
    proj_row$var_id <- var_id

    ## Select the linear system for model fitting:
    var_lin_system <- lin_systems[lin_systems$var_id==var_id,]

    model_conf <- data.frame(model_config_df)

    ## Need to know whether vectorisation is enabled, as depending on 
    ## the architecture it can cause execution ports to fuse:
    var_lin_system_var_split <- split_col(var_lin_system, "var_id")
    if (!("SIMD.len" %in% names(var_lin_system_var_split))) {
        ## Assume that SIMD was disabled.
        simd_len = 1
    } else {
        simd_len = var_lin_system_var_split$SIMD.len[1]
    }
    isa = var_lin_system_var_split$Instruction.set[1]
    if (simd_len > 1) {
        if (isa == "AVX512") {
            model_conf <- rbind(model_conf, data.frame("key"="avx512_simd_enabled", "value"=TRUE))
        }
    }
    if (model_fitting_strategy == model_fitting_strategy_datums[["miniDifferences"]]) {
        model_conf <- rbind(model_conf, data.frame("key"="predict_perf_diff", "value"=TRUE))
    }
    model_conf <- model_conf[model_conf$key!="baseline_kernel",]
    model_conf <- model_conf[model_conf$key!="model_fitting_strategy",]

    ## Prepare and write out model fitting data for Python:
    model_coef_colnames <- model_eu_and_mem_colnames
    model_data <- var_lin_system[,c("Flux.variant", "wg_cycles", model_coef_colnames)]
    model_fitting_data <- model_data[model_data$Flux.variant!="Flux-Synthetic",]
    model_fitting_data[,"wg_cycles"] <- round(model_fitting_data[,"wg_cycles"], digits=1)

    ## Prune unnecessary columns to assist model fitting:
    # Drop zero columns:
    for (cn in intersect(exec_unit_colnames, names(model_fitting_data))) {
        if (sum(model_fitting_data[,cn]) == 0 ) {
            model_fitting_data[,cn] <- NULL
        }
    }
    for (cn in intersect(mem_event_colnames, names(model_fitting_data))) {
        if (cn == "mem.loads" && (do_load_penalty || do_spill_penalty)) {
            ## Keep a load coefficient for these penalties.
            next
        }
        if (sum(model_fitting_data[,cn]) == 0 ) {
            model_fitting_data[,cn] <- NULL
        }
    }
    # Drop spills column if not used by model
    if (!do_spill_penalty && do_ignore_loads_stores) {
        if ("mem.spills" %in% names(model_fitting_data)) {
            model_fitting_data[,"mem.spills"] <- NULL
        }
    }

    model_dirname <- paste0("Modelling", Sys.getpid())

    if (!dir.exists(model_dirname)) {
        dir.create(model_dirname)
    } else {
        do.call(file.remove, list(list.files(paste0(model_dirname,"/"), full.names = TRUE)))
    }
    write.csv(model_fitting_data, file.path(model_dirname, "fitting_data.csv"), row.names=FALSE)
    write.csv(model_conf, file.path(model_dirname, "insn_model_conf.csv"), row.names=FALSE)

    ## Run Python solver to estimate CPIs:
    python_output <- system(paste("python", file.path(script_dirpath, "model_interface.py"), "-f", "-d", model_dirname), intern=TRUE)
    solution_filepath <- file.path(model_dirname, "solution.csv")
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
    }

    if (debug_model_fitting) {
        # Generate WG predictions to assist debugging:
        mg_cfd_data <- var_lin_system
        mg_cfd_data <- mg_cfd_data[,c("wg_cycles", model_coef_colnames)]
        mg_cfd_data$wg_cycles <- 0.0
        for (mcc in model_coef_colnames) {
            if (sum(mg_cfd_data[,mcc]) == 0) {
                mg_cfd_data[,mcc] <- NULL
            }
        }
        write.csv(mg_cfd_data, file.path(model_dirname, "prediction_data.csv"), row.names=FALSE)
        python_output <- system(paste("python", file.path(script_dirpath, "model_interface.py"), "-p", "-d", model_dirname), intern=TRUE)
        prediction_filepath <- file.path(model_dirname, "prediction.csv")
        if (!file.exists(prediction_filepath)) {
            print(paste0("ERROR: Python script did not complete and write prediction to csv: ", prediction_filepath))
            print(python_output)
            q()
        }
        prediction <- read.csv(prediction_filepath)
        model_data <- cbind(model_data, prediction)
        write.csv(model_data, file.path(model_dirname, "modelling_input_data.csv"), row.names=FALSE)
    } else {
        ## Cleanup:
        system(paste("rm -r", model_dirname))
    }

    ll = lock(lock_filename)
    if (!is.locked(ll)) {
        stop("Failed to obtain file lock")
    }
    append_and_write_row(proj_row)
    unlock(ll)

    return(proj_row)
}

prune_model_perms <- function(model_perms) {
    model_perms_pruned <- NULL

    for (model_perm_id in 1:nrow(model_perms)) {
        # Decode model_perm_id:
        model_perm <- model_perms[model_perm_id,]
        var_id <- as.character(model_perm$var_id)

        baseline_kernel <- as.character(model_perm$baseline_kernel)
        model_fitting_strategy <- as.character(model_perm$model_fitting_strategy)
        do_spill_penalty <- model_perm$do_spill_penalty
        do_load_penalty <- model_perm$do_load_penalty
        do_prune_insn_classes <- model_perm$do_prune_insn_classes
        do_ignore_loads_stores <- model_perm$do_ignore_loads_stores
        optimisation_search_algorithm <- model_perm$optimisation_search_algorithm
        opt_num_iters <- model_perm$opt_num_iters
        opt_num_jumps <- model_perm$opt_num_jumps
        opt_step_size <- model_perm$opt_step_size

        if (optimisation_search_algorithm=="basin") {
            model_perm$optimisation_search_option <- paste0("iters=",opt_num_iters,"^jumps=",opt_num_jumps,"^steps=",opt_step_size)
            model_perm$optimisation_search_cost   <- opt_num_iters*opt_num_jumps
        } else {
            model_perm$optimisation_search_option <- ""
            model_perm$optimisation_search_cost   <- 1
        }

        # ll = lock(lock_filename)
        # if (!is.locked(ll)) {
        #     stop("Failed to obtain file lock")
        # }
        v <- cpi_estimate_already_exists(var_id, model_perm)
        # unlock(ll)
        if (!v) {
            if (is.null(model_perms_pruned)) {
                model_perms_pruned <- model_perms[model_perm_id,]
            } else {
                model_perms_pruned <- safe_rbind(model_perms_pruned, model_perms[model_perm_id,])
            }
        }
    }

    return(model_perms_pruned)
}

model_perms <- prune_model_perms(model_perms)

model_perms_ids <- 1:nrow(model_perms)
## Use this loop for debugging model fitting:
# for (i in model_perms_ids) {
#     print(paste("i=",i))
#     r <- predict_fn(i)
# }
# q()
res <- mclapply(model_perms_ids, predict_fn, mc.cores = numCores)
# Reload 'cpi_estimates' from filesystem, only way I know of to synchronise between worker processes:
projections_filename <- "cpi_estimates.csv"
if (file.exists(projections_filename)) {
    cpi_estimates <- read.csv(projections_filename, stringsAsFactors=FALSE)
    cpi_estimates[is.na(cpi_estimates[,"optimisation_search_option"]), "optimisation_search_option"] <- ""
} else {
    cpi_estimates <- NULL
}

print("Finished fitting, now writing out.")

model_coefs <- c()
for (col in names(cpi_estimates)) {
    if (startsWith(col, "eu.") || startsWith(col, "mem.")) {
        model_coefs <- c(model_coefs, col)
    }
}

model_conf_params <- intersect(model_conf_params, names(cpi_estimates))
order_cols <- c()
if ("var_id" %in% names(cpi_estimates)) {
    order_cols <- c(order_cols, "var_id")
    # cols_before <- names(cpi_estimates)
    # cpi_estimates <- split_col(cpi_estimates, "var_id")
    # new_cols <- setdiff(names(cpi_estimates), cols_before)
    # order_cols <- c(order_cols, new_cols)
}
if (length(model_conf_params) > 0) {
    order_cols <- c(order_cols, model_conf_params)
}
if ("optimisation_search_cost" %in% names(cpi_estimates)) {
    order_cols <- c(order_cols, "optimisation_search_cost")
}
if ("optimisation_search_option" %in% names(cpi_estimates)) {
    # order_cols <- c(order_cols, "optimisation_search_option")
    cpi_estimates <- split_col(cpi_estimates, "optimisation_search_option")
}
if (length(order_cols) > 0) {
    if (length(order_cols) == 1) {
        cpi_estimates <- cpi_estimates[order(cpi_estimates[,order_cols[1]]),]
    } else {
        cpi_estimates <- cpi_estimates[do.call(order, cpi_estimates[,order_cols]),]
    }
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

print(paste("CPI estimates written to", projections_filename))
write.csv(cpi_estimates, projections_filename, row.names=FALSE)

# Drop non-varying columns:
for (p in names(cpi_estimates)) {
    if (length(unique(cpi_estimates[,p]))==1) {
        cpi_estimates[,p] <- NULL
    }
}
projections_cleaned_filename <- "cpi_estimates.cleaned.csv"
write.csv(cpi_estimates, projections_cleaned_filename, row.names=FALSE)

## Cleanup:
# system(paste("rm -r", "Modelling"))
