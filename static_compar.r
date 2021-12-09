library("tidyverse")
library("scmamp")
# path
parent_path <- fs::path_wd()
result_paths <- fs::dir_ls(path = fs::path(parent_path, "result"),
    regexp = "Scores")
# read socres file
result_all <- purrr::map(result_paths, readr::read_csv)
result_f1 <- as_tibble(purrr::map(result_all, purrr::chuck("F1")))
result_auc <- as_tibble(purrr::map(result_all, purrr::chuck("AUC")))
result_h <- as_tibble(purrr::map(result_all, purrr::chuck("H-measure")))
result_ks <- as_tibble(purrr::map(result_all, purrr::chuck("KS_score")))
result_brier <- as_tibble(purrr::map(result_all, purrr::chuck("Brier_score")))
result_log <- as_tibble(purrr::map(result_all, purrr::chuck("Log_loss")))
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
result_1 <- scmamp::imanDavenportTest(result_f1)
result_2 <- scmamp::postHocTest(result_f1, 
    test = "aligned ranks", correct = "rom")
scmamp::plotRanking(pvalues = result_2$corrected.pval, 
    summary = result_2$summary, alpha = 0.05)