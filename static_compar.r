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
regexr_body <- rlang::expr(str_match(text,
    pattern = "(?!.*/)(.*)(?:\\sScores)")[, 2])
regexr <- rlang::new_function(rlang::pairlist2(text = list()), regexr_body)
result_f1 <- dplyr::rename_with(result_f1, .fn = regexr)
result_auc <- dplyr::rename_with(result_auc, .fn = regexr)
result_h <- dplyr::rename_with(result_h, .fn = regexr)
result_ks <- dplyr::rename_with(result_ks, .fn = regexr)
result_brier <- dplyr::rename_with(result_brier, .fn = regexr)
result_log <- dplyr::rename_with(result_log, .fn = regexr)
# statistical testing
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
# Ploting
## cp /tmp/Rtmp0mjEzB/vscode-R/plot.png /workspaces/Project/
plot_f1 <- scmamp::plotRanking(pvalues = pairwise_f1$corrected.pval,
    summary = pairwise_f1$summary, alpha = 0.05)
plot_auc <- scmamp::plotRanking(pvalues = pairwise_auc$corrected.pval,
    summary = pairwise_auc$summary, alpha = 0.05)
plot_h <- scmamp::plotRanking(pvalues = pairwise_h$corrected.pval,
    summary = pairwise_h$summary, alpha = 0.05)
plot_ks <- scmamp::plotRanking(pvalues = pairwise_ks$corrected.pval,
    summary = pairwise_ks$summary, alpha = 0.05)
plot_brier <- scmamp::plotRanking(pvalues = pairwise_brier$corrected.pval,
    summary = pairwise_brier$summary, alpha = 0.05)
plot_log <- scmamp::plotRanking(pvalues = pairwise_log$corrected.pval,
    summary = pairwise_log$summary, alpha = 0.05)
