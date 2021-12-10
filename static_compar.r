library("tidyverse")
library("scmamp")
# path
parent_path <- fs::path_wd()
result_paths <- fs::dir_ls(path = fs::path(parent_path, "result"),
    regexp = "Scores")
# read socres file
result_all <- purrr::map(result_paths, readr::read_csv)
result_f1 <- tibble::as_tibble(purrr::map(result_all, purrr::chuck("F1")))
result_auc <- tibble::as_tibble(purrr::map(result_all, purrr::chuck("AUC")))
result_h <- tibble::as_tibble(purrr::map(result_all, purrr::chuck("H-measure")))
result_ks <- tibble::as_tibble(purrr::map(result_all, purrr::chuck("KS_score")))
result_brier <- tibble::as_tibble(purrr::map(result_all,
    purrr::chuck("Brier_score")))
result_log <- tibble::as_tibble(purrr::map(result_all,
    purrr::chuck("Log_loss")))
# rename columns
## Regular Expression
regexr_body <- rlang::expr(stringr::str_match(text,
    pattern = "(?!.*/)(.*)(?:\\sScores)")[, 2])
regexr <- rlang::new_function(rlang::pairlist2(text = list()), regexr_body)
## rename
result_f1 <- dplyr::rename_with(result_f1, .fn = regexr)
result_auc <- dplyr::rename_with(result_auc, .fn = regexr)
result_h <- dplyr::rename_with(result_h, .fn = regexr)
result_ks <- dplyr::rename_with(result_ks, .fn = regexr)
result_brier <- dplyr::rename_with(result_brier, .fn = regexr)
result_log <- dplyr::rename_with(result_log, .fn = regexr)
# Ploting
plot_body <- rlang::expr(scmamp::plotRanking(pvalues = result$corrected.pval,
    summary = result$summary, alpha = 0.05))
rank_plot <- rlang::new_function(rlang::pairlist2(result = NULL), plot_body)
# statistical testing all
## F1
omnibus_f1 <- scmamp::imanDavenportTest(result_f1)
pairwise_f1 <- scmamp::postHocTest(result_f1,
    test = "aligned ranks", correct = "rom")
pairwise_result_f1 <- as_tibble(pairwise_f1$corrected.pval, rownames = NA)
## AUC
omnibus_auc <- scmamp::imanDavenportTest(result_auc)
pairwise_auc <- scmamp::postHocTest(result_auc,
    test = "aligned ranks", correct = "rom")
pairwise_result_auc <- as_tibble(pairwise_auc$corrected.pval, rownames = NA)
## H-measure
omnibus_h <- scmamp::imanDavenportTest(result_h)
pairwise_h <- scmamp::postHocTest(result_h,
    test = "aligned ranks", correct = "rom")
pairwise_result_h <- as_tibble(pairwise_h$corrected.pval, rownames = NA)
## KS_score
omnibus_ks <- scmamp::imanDavenportTest(result_ks)
pairwise_ks <- scmamp::postHocTest(result_ks,
    test = "aligned ranks", correct = "rom")
pairwise_result_ks <- as_tibble(pairwise_ks$corrected.pval, rownames = NA)
## Brier_score
omnibus_brier <- scmamp::imanDavenportTest(result_brier)
pairwise_brier <- scmamp::postHocTest(result_brier,
    test = "aligned ranks", correct = "rom")
pairwise_result_brier <- as_tibble(pairwise_brier$corrected.pval, rownames = NA)
## Log_loss
omnibus_log <- scmamp::imanDavenportTest(result_log)
pairwise_log <- scmamp::postHocTest(result_log,
    test = "aligned ranks", correct = "rom")
pairwise_result_log <- as_tibble(pairwise_log$corrected.pval, rownames = NA)
# statistical testing non SMOTE
non_smote <- stringr::str_subset(attributes(result_f1)$names,
        pattern = "^.*(?=\\swith\\sSMOTE)", negate = TRUE)
## f1 non SMOTE
omnibus_f1_ns <- scmamp::imanDavenportTest(result_f1[non_smote])
pairwise_f1_ns <- scmamp::postHocTest(result_f1[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_f1_ns <- as_tibble(pairwise_f1_ns$corrected.pval, rownames = NA)
## AUC non SMOTE
omnibus_auc_ns <- scmamp::imanDavenportTest(result_auc[non_smote])
pairwise_auc_ns <- scmamp::postHocTest(result_auc[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_auc_ns <- as_tibble(pairwise_auc_ns$corrected.pval,
    rownames = NA)
## H-measure non SMOTE
omnibus_h_ns <- scmamp::imanDavenportTest(result_h[non_smote])
pairwise_h_ns <- scmamp::postHocTest(result_h[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_h_ns <- as_tibble(pairwise_h_ns$corrected.pval, rownames = NA)
## KS_score non SMOTE
omnibus_ks_ns <- scmamp::imanDavenportTest(result_ks[non_smote])
pairwise_ks_ns <- scmamp::postHocTest(result_ks[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_ks_ns <- as_tibble(pairwise_ks_ns$corrected.pval, rownames = NA)
## Brier_score non SMOTE
omnibus_brier_ns <- scmamp::imanDavenportTest(result_brier[non_smote])
pairwise_brier_ns <- scmamp::postHocTest(result_brier[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_brier_ns <- as_tibble(pairwise_brier_ns$corrected.pval,
    rownames = NA)
## Log_loss non SMOTE
omnibus_log_ns <- scmamp::imanDavenportTest(result_log[non_smote])
pairwise_log_ns <- scmamp::postHocTest(result_log[non_smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_log_ns <- as_tibble(pairwise_log_ns$corrected.pval,
    rownames = NA)
# statistical testing SMOTE
smote <- stringr::str_subset(attributes(result_f1)$names,
        pattern = "^.*(?=\\swith\\sSMOTE)")
## F1 SMOTE
omnibus_f1_s <- scmamp::imanDavenportTest(result_f1[smote])
pairwise_f1_s <- scmamp::postHocTest(result_f1[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_f1_s <- as_tibble(pairwise_f1_s$corrected.pval, rownames = NA)
## AUC SMOTE
omnibus_auc_s <- scmamp::imanDavenportTest(result_auc[smote])
pairwise_auc_s <- scmamp::postHocTest(result_auc[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_auc_s <- as_tibble(pairwise_auc_s$corrected.pval, rownames = NA)
## H-measure SMOTE
omnibus_h_s <- scmamp::imanDavenportTest(result_h[smote])
pairwise_h_s <- scmamp::postHocTest(result_h[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_h_s <- as_tibble(pairwise_h_s$corrected.pval, rownames = NA)
## KS_score SMOTE
omnibus_ks_s <- scmamp::imanDavenportTest(result_ks[smote])
pairwise_ks_s <- scmamp::postHocTest(result_ks[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_ks_s <- as_tibble(pairwise_ks_s$corrected.pval, rownames = NA)
## Brier_score SMOTE
omnibus_brier_s <- scmamp::imanDavenportTest(result_brier[smote])
pairwise_brier_s <- scmamp::postHocTest(result_brier[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_brier_s <- as_tibble(pairwise_brier_s$corrected.pval,
    rownames = NA)
## Log_loss SMOTE
omnibus_log_s <- scmamp::imanDavenportTest(result_log[smote])
pairwise_log_s <- scmamp::postHocTest(result_log[smote],
    test = "aligned ranks", correct = "rom")
pairwise_result_log_s <- as_tibble(pairwise_log_s$corrected.pval,
    rownames = NA)
