library(reshape)


var_id_sep = "^"

get_data_col_names <- function(d) {
    data_classes <- c('flux.update', 'flux', 'update', 'compute_step', 'time_step', 'restrict', 'prolong', 'indirect_rw')
    
    possible_data_col_names <- c()
    for (t in data_classes) {
        for (l in seq(0,3)) {
            tl <- paste0(t,l)
            possible_data_col_names <- c(possible_data_col_names, tl)
        }
        tl <- paste0(t, ".total")
        possible_data_col_names <- c(possible_data_col_names, tl)
    }
    possible_data_col_names <- c(possible_data_col_names, "Total")

    present_data_col_names <- intersect(names(d), possible_data_col_names)

    for (cn in names(d)) {
        if (startsWith(cn, "insn.") || startsWith(cn, "eu.") || startsWith(cn, "mem.")) {
            present_data_col_names <- c(present_data_col_names, cn)
        }
    }

    return(present_data_col_names)
}

get_var_col_names <- function(d, drop_nonvarying=FALSE) {
    data_col_names <- get_data_col_names(d)
    var_col_names <- setdiff(names(d), data_col_names)

    for (var_col in var_col_names) {
        var_col_unique_values <- unique(d[,var_col])
        if (drop_nonvarying) {
            if (length(var_col_unique_values)==1 && is.na(var_col_unique_values[1])) {
                var_col_names <- setdiff(var_col_names, var_col)
            }
        }
    }

    return(var_col_names)
}

rename_col <- function(D, oldname, newname) {
    if (!(oldname %in% names(D))) {
        # print(paste0("WARNING: Column '", oldname, "' not in data.frame, cannot rename it."))
        stop(paste0("ERROR: Rename of column '", oldname, "' -> '", newname, "' failed, not in data.frame."))
        return(D)
    }

    if (newname %in% names(D)) {
        print(paste0("ERROR: '", newname, "' already in data.frame, cannot rename '", oldname, "' to it."))
        q()
    }

    D[,newname] <- D[,oldname]
    D[,oldname] <- NULL
    return(D)
}

concat_cols <- function(D, cols_to_concat, new_col_name, delete_cols) {
    if (nrow(D)==0) {
        stop("ERROR: concat_cols() called on empty data frame")
    }

    if (new_col_name %in% names(D)) {
        D[,new_col_name] <- NULL
    }

    cols_to_concat <- sort(intersect(names(D), cols_to_concat))
    if (length(cols_to_concat) > 0) {
        D[,new_col_name] <- paste0(cols_to_concat[1],"=",D[,cols_to_concat[1]])
        if (delete_cols) {
            D[,cols_to_concat[1]] <- NULL
        }
        variable_col_names_remainder <- setdiff(cols_to_concat, cols_to_concat[1])
        for (var in variable_col_names_remainder) {
            D[,new_col_name] <- paste0(D[,new_col_name], var_id_sep, var,"=",D[,var])
            if (delete_cols) {
                D[,var] <- NULL
            }
        }
        D[,new_col_name] <- as.character(D[,new_col_name])
    } else {
        D[,new_col_name] <- "No_cols"
    }

    return(D)
}

split_col <- function(D, col_to_split) {
    if (!(col_to_split %in% names(D))) {
        return(D)
    }

    values <- unique(D[,col_to_split])
    if (length(values)==1 && values[1]=="No_cols") {
        D[,col_to_split] <- NULL
        return(D)
    }

    col_decoded <- read.table(text=as.character(D[,col_to_split]), sep=var_id_sep, colClasses="character")
    for (v in names(col_decoded)) {
        # print(paste("Decoding", v))

        if (v %in% names(D)) {
            # print(paste0("WARNING from split_col(): '", v, "' already present in D"))
            col_decoded[,v] <- NULL
            next
        }

        v_name <- strsplit(col_decoded[1,v], "=")[[1]][1]
        v_values <- unlist(lapply(strsplit(col_decoded[,v], "="), function(x) if (length(x)>1) { x[[2]] } else { "" } ))
        col_decoded[,v_name] <- v_values
        col_decoded[,v] <- NULL
    }
    D <- cbind(col_decoded, D)
    D[,col_to_split] <- NULL

    return(D)
}


infer_run_value <- function(var, src_perf_data_dir) {
    if (!(dir.exists(src_perf_data_dir))) {
        stop(paste0("infer_run_value(var=", var, ") called on non-existent directory: ", src_perf_data_dir))
    }
    files <- list.files(path=src_perf_data_dir, pattern=".*.csv", full.names=TRUE, recursive=TRUE)
    if (length(files)==0) {
        stop(paste("No csv files found anywhere in directory:", src_perf_data_dir))
    }
    sample_data <- NULL
    var_found <- FALSE
    for (i in 1:length(files)) {
        sample_data <- read.csv(files[i])
        if (var %in% names(sample_data)) {
            var_found <- TRUE
            break
        }
    }
    if (!var_found) {
        if (var == "SIMD.len") {
            return(1)
        }

        stop(paste0("Column '", var, "' not found"))
    }

    var_values <- unique(sample_data[,var])
    if (length(var_values) > 1) {
        stop(paste0(length(var_values), " values found for column ", var))
    }

    return(sample_data[1,var])
}


safe_rbind <- function(d1, d2) {
    n_d1 <- names(d1)
    n_d2 <- names(d2)
    if (length(setdiff(n_d1, n_d2)) > 0) {
        print("ERROR: RHS of rbind() missing columns:")
        print(setdiff(n_d1, n_d2))
        q()
    }
    if (length(setdiff(n_d2, n_d1)) > 0) {
        print("ERROR: LHS of rbind() missing columns:")
        print(setdiff(n_d2, n_d1))
        q()
    }
    d <- rbind(d1, d2)
    n_d <- names(d)
    if (length(setdiff(n_d, n_d1)) > 0) {
        print("ERROR: rbind() has added columns to LHS:")
        print(setdiff(n_d, n_d1))
        q()
    }

    return(d)
}

safe_merge <- function(d1, d2) {
    nrows_pre <- nrow(d1)
    dm <- merge(d1, d2)
    nrows_post <- nrow(dm)
    if (nrows_post != nrows_pre) {
        print(paste(nrows_pre, "rows before safe_merge() but", nrows_post, "rows after"))
        n1 <- names(d1)
        n2 <- names(d2)
        if (length(setdiff(n1,n2))>0) {
            print("  rhs missing these columns:")
            print(setdiff(n1,n2))
        }
        if (length(setdiff(n2,n1))>0) {
            print("  lhs missing these columns:")
            print(setdiff(n2,n1))
        }
        q()
    }
    return(dm)
}

process_kmp_hw_subset <- function(d) {
    if ("KMP.hw.subset" %in% names(d)) {
        d$KMP.hw.subset <- as.character(d$KMP.hw.subset)

        if ("KMP.affinity" %in% names(d)) {
            d$KMP.affinity <- as.character(d$KMP.affinity)
        } else {
            d$KMP.affinity <- ""
        }

        filter <- d$KMP.hw.subset != ""

        num.smt <- rep(1, length(filter))
        num.sockets <- rep(1, length(filter))
        for (i in 2:16) {
            num.smt[filter & grepl(paste0(i,"t"), d$KMP.hw.subset)] <- i
            num.sockets[filter & grepl(paste0(i,"s"), d$KMP.hw.subset)] <- i
        }

        smt_vals <- unique(num.smt)
        skt_vals <- unique(num.sockets)

        if ( (length(smt_vals) > 1) && (length(skt_vals) > 1) ) {
            d[filter, "KMP.affinity"] <- paste0("explicit",var_id_sep,"smt=",num.smt[filter],var_id_sep,"sk=",num.sockets[filter])
        }
        else if (length(smt_vals) > 1) {
            d[filter, "KMP.affinity"] <- paste0("explicit",var_id_sep,"smt=",num.smt[filter])
        }
        else if (length(skt_vals) > 1) {
            d[filter, "KMP.affinity"] <- paste0("explicit",var_id_sep,"sk=",num.sockets[filter])
        }
        else {
            d[filter, "KMP.affinity"] <- "explicit"
        }
        # d[filter, "KMP.affinity"] <- paste0("explicit;smt=",num.smt[filter],";sk=",num.sockets[filter])
        
        d$KMP.hw.subset <- NULL
    }
    return (d)
}

split_kmp_hw_subset <- function(d) {
    if ("KMP.hw.subset" %in% names(d)) {
        d$KMP.hw.subset <- as.character(d$KMP.hw.subset)

        if ("KMP.affinity" %in% names(d)) {
            d$KMP.affinity <- as.character(d$KMP.affinity)
        } else {
            d$KMP.affinity <- ""
        }

        filter <- d$KMP.hw.subset != ""

        num.smt <- rep(1, length(filter))
        num.sockets <- rep(1, length(filter))
        for (i in 2:16) {
            num.smt[filter & grepl(paste0(i,"t"), d$KMP.hw.subset)] <- i
            num.sockets[filter & grepl(paste0(i,"s"), d$KMP.hw.subset)] <- i
        }

        smt_vals <- unique(num.smt)
        skt_vals <- unique(num.sockets)

        d$SKT <- num.sockets
        d$SMT <- num.smt
        
        # d$KMP.hw.subset <- NULL
    }
    return (d)
}

reshape_data_cols <- function(D, kernels_to_retain) {
    id_var_candidates <- c("iflux.variant", "Flux.variant", "var_id", "PAPI.counter", "Num.threads", "Instruction.set", "CC")
    id_var_candidates <- c(id_var_candidates, "SIMD.conflict.avoidance.strategy")
    id_vars <- intersect(id_var_candidates, names(D))
    data_vars <- setdiff(names(D), id_vars)
    if (length(data_vars) == 0) {
        stop("reshape_data_cols() called on DF with no data cols to melt.")
    }
    D <- melt(D, id.vars=id_vars)
    names(D) <- c(id_vars, "loop", "value")
    D$kernel <- ""
    D$level <- -1
    loop_values <- unique(D$loop)
    for (l in 0:3) {
        for (k in kernels_to_retain) {
            kl <- paste0(k, l)
            if (kl %in% loop_values) {
                D[D$loop==kl,"kernel"] <- k
                D[D$loop==kl,"level"] <- l
            }
        }
    }
    # if (sum(D$level==-1) > 0) {
    #     print(D[D$level==-1,][1,])
    #     stop(paste("failed to assign level to all reshaped entries"))
    # }
    D <- D[D$level != -1,]
    D$loop <- NULL

    return(D)
}

