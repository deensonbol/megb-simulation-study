library(MEGB)
library(gbm)
library(ggplot2)

# ============================================================
# PART A: Single-run sanity check (training fit only)
# Purpose: Confirm MEGB runs and produces reasonable output
# ============================================================

set.seed(1)

data <- simLong(
  n = 20, 
  p = 170,
  rel_p = 6, 
  time_points = 10, 
  rho_W = 0.6, 
  rho_Z = 0.6, 
  random_sd_intercept = sqrt(0.5), 
  random_sd_slope = sqrt(3), 
  noise_sd = 0.5, 
  linear = TRUE
)

megb <- MEGB(
  X = as.matrix(data[, -1:-5]), 
  Y = as.matrix(data$Y), 
  Z = as.matrix(data[, 4:5]), 
  id = data$id, 
  time = data$time, 
  ntree = 500, 
  cv.folds = 0, 
  verbose = FALSE
)

pred <- predict(
  megb, 
  X = as.matrix(data[, -1:-5]), 
  Z = as.matrix(data[, 4:5]), 
  id = data$id, 
  time = data$time, 
  ntree = 500
)

# Training RMSE (sanity check only)
sqrt(mean((data$Y - pred)^2))



# ============================================================
# PART B: Simulation 1 - Subject-level holdout
# Purpose: Evaluate generalization to new individuals (new subjects)
# ============================================================

run_one <- function(seed, n_subj = 20, p = 170, ntree = 500, test_frac = 0.2) {
  
  tryCatch({
    
    set.seed(seed)
    
    data <- simLong(
      n = n_subj, 
      p = p, 
      rel_p = 6, 
      time_points = 10,
      rho_W = 0.6, 
      rho_Z = 0.6,
      random_sd_intercept = sqrt(0.5),
      random_sd_slope     = sqrt(3),
      noise_sd = 0.5,
      linear = TRUE
    )
    
    # ---- hold out SUBJECTS (new-subject prediction) ----
    ids <- unique(data$id)
    set.seed(seed + 1000)
    test_ids <- sample(ids, size = ceiling(test_frac * length(ids)))
    train <- data[!(data$id %in% test_ids), ]
    test  <- data[(data$id %in% test_ids), ]
    
    X_train_df <- as.data.frame(train[, -1:-5])
    X_test_df  <- as.data.frame(test[, -1:-5])
    
    # ---- MEGB ----
    fit <- MEGB(
      X    = as.matrix(X_train_df),
      Y    = as.matrix(train$Y),
      Z    = as.matrix(train[, 4:5]),
      id   = train$id,
      time = train$time,
      ntree = ntree,
      cv.folds = 0,
      verbose = FALSE
    )
    
    if (is.null(fit$forest)) {
      return(data.frame(seed = seed, n = n_subj,
                        rmse_megb = NA_real_, rmse_gbm = NA_real_,
                        status = "forest_NULL"))
    }
    
    # fixed-only prediction for new subjects
    pred_megb <- predict(fit$forest, newdata = X_test_df, n.trees = ntree)
    rmse_megb <- sqrt(mean((test$Y - pred_megb)^2))
    
    # ---- GBM baseline ----
    gbm_train <- data.frame(Y = train$Y, X_train_df)
    fit_gbm <- gbm(
      Y ~ .,
      data = gbm_train,
      distribution = "gaussian",
      n.trees = ntree,
      interaction.depth = 3,
      shrinkage = 0.05,
      n.minobsinnode = 2,
      bag.fraction = 0.8,
      verbose = FALSE
    )
    
    pred_gbm <- predict(fit_gbm, newdata = X_test_df, n.trees = ntree)
    rmse_gbm <- sqrt(mean((test$Y - pred_gbm)^2))
    
    data.frame(seed = seed, n = n_subj,
               rmse_megb = rmse_megb, rmse_gbm = rmse_gbm,
               status = "ok")
    
  }, error = function(e) {
    data.frame(seed = seed, n = n_subj,
               rmse_megb = NA_real_, rmse_gbm = NA_real_,
               status = paste0("error: ", e$message))
  })
}


# --------------------------------------------------
# Small test run
# --------------------------------------------------
seeds <- 1:5
ns <- c(10, 20)

res_small <- do.call(rbind, lapply(ns, function(n0) {
  do.call(rbind, lapply(seeds, function(s) run_one(seed = s, n_subj = n0)))
}))

print(res_small)
table(res_small$status)


# --------------------------------------------------
# Larger experiment
# --------------------------------------------------
seeds <- 1:200
ns <- c(20, 50, 100)

res <- do.call(rbind, lapply(ns, function(n0) {
  do.call(rbind, lapply(seeds, function(s) run_one(seed = s, n_subj = n0)))
}))

table(res$status)

# Summary table: mean, sd, and number of successful runs
summary_tbl <- aggregate(cbind(rmse_megb, rmse_gbm) ~ n,
                         data = res,
                         FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                             sd = sd(x, na.rm = TRUE),
                                             n_ok = sum(!is.na(x))))
summary_tbl



# ============================================================
# PART C: Simulation 2 — Time-split evaluation
# Purpose: Predict future time points within same subjects
# ============================================================

run_one_time_split <- function(seed, n_subj = 20, p = 170,
                               ntree = 500, test_last_k = 2) {
  
  tryCatch({
    
    set.seed(seed)
    
    data <- simLong(
      n = n_subj,
      p = p,
      rel_p = 6,
      time_points = 10,
      rho_W = 0.6,
      rho_Z = 0.6,
      random_sd_intercept = sqrt(0.5),
      random_sd_slope     = sqrt(3),
      noise_sd = 0.5,
      linear = TRUE
    )
    
    # Ensure ordered by subject/time
    data <- data[order(data$id, data$time), ]
    
    # Train = all but last k times per subject; Test = last k per subject
    is_train <- ave(data$time, data$id, FUN = function(t) rank(t) <= (length(t) - test_last_k))
    train <- data[is_train == 1, ]
    test  <- data[is_train == 0, ]
    
    X_train_df <- as.data.frame(train[, -1:-5])
    X_test_df  <- as.data.frame(test[, -1:-5])
    
    # ---- MEGB ---- (same subjects: can use full predict(fit, ...) )
    fit <- MEGB(
      X    = as.matrix(X_train_df),
      Y    = as.matrix(train$Y),
      Z    = as.matrix(train[, 4:5]),
      id   = train$id,
      time = train$time,
      ntree = ntree,
      cv.folds = 0,
      verbose = FALSE
    )
    
    if (is.null(fit$forest)) {
      return(data.frame(seed = seed, n = n_subj,
                        rmse_megb = NA_real_, rmse_gbm = NA_real_,
                        status = "forest_NULL"))
    }
    
    pred_megb <- predict(
      fit,
      X    = as.matrix(X_test_df),
      Z    = as.matrix(test[, 4:5]),
      id   = test$id,
      time = test$time,
      ntree = ntree
    )
    rmse_megb <- sqrt(mean((test$Y - pred_megb)^2))
    
    # ---- GBM baseline ---- (ignores correlation; just predicts from X)
    gbm_train <- data.frame(Y = train$Y, X_train_df)
    fit_gbm <- gbm(
      Y ~ .,
      data = gbm_train,
      distribution = "gaussian",
      n.trees = ntree,
      interaction.depth = 3,
      shrinkage = 0.05,
      n.minobsinnode = 2,
      bag.fraction = 0.8,
      verbose = FALSE
    )
    pred_gbm <- predict(fit_gbm, newdata = X_test_df, n.trees = ntree)
    rmse_gbm <- sqrt(mean((test$Y - pred_gbm)^2))
    
    data.frame(seed = seed, n = n_subj,
               rmse_megb = rmse_megb, rmse_gbm = rmse_gbm,
               status = "ok")
    
  }, error = function(e) {
    data.frame(seed = seed, n = n_subj,
               rmse_megb = NA_real_, rmse_gbm = NA_real_,
               status = paste0("error: ", e$message))
  })
}


# --------------------------------------------------
# Run time-split experiment
# --------------------------------------------------
seeds <- 1:200
ns <- c(20, 50, 100)

resB <- do.call(rbind, lapply(ns, function(n0) {
  do.call(rbind, lapply(seeds, function(s) run_one_time_split(seed = s, n_subj = n0, test_last_k = 2)))
}))

table(resB$status)

summaryB <- aggregate(cbind(rmse_megb, rmse_gbm) ~ n, data = resB,
                      FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                          sd = sd(x, na.rm = TRUE),
                                          n_ok = sum(!is.na(x))))
summaryB

# difference table (MEGB - GBM)
resB$diff <- resB$rmse_megb - resB$rmse_gbm
diffB <- aggregate(diff ~ n, data = resB,
                   FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                       sd   = sd(x, na.rm = TRUE),
                                       n_ok = sum(!is.na(x))))
diffB



# ============================================================
# PART D: Plots (based on time-split results)
# ============================================================

summaryB_clean <- data.frame(
  n         = summaryB$n,
  megb_mean = summaryB$rmse_megb[, "mean"],
  megb_sd   = summaryB$rmse_megb[, "sd"],
  gbm_mean  = summaryB$rmse_gbm[, "mean"],
  gbm_sd    = summaryB$rmse_gbm[, "sd"]
)

plot_df <- rbind(
  data.frame(n = summaryB_clean$n, rmse = summaryB_clean$megb_mean, method = "MEGB"),
  data.frame(n = summaryB_clean$n, rmse = summaryB_clean$gbm_mean,  method = "GBM")
)

# Plot 1: Mean RMSE vs n (MEGB vs GBM)
ggplot(plot_df, aes(x = n, y = rmse, color = method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = summaryB_clean$n) +
  labs(
    title = "Time-Split Prediction Error by Sample Size",
    x = "Number of Subjects (n)",
    y = "Mean RMSE",
    color = "Method"
  ) +
  theme_minimal(base_size = 14)

# Plot 2: Distribution of RMSE Differences (MEGB − GBM) by n
ggplot(resB, aes(x = factor(n), y = diff)) +
  geom_boxplot(fill = "#A6CEE3", alpha = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.8) +
  labs(
    title = "Difference in Prediction Error (MEGB − GBM)",
    x = "Number of Subjects (n)",
    y = "RMSE Difference"
  ) +
  theme_minimal(base_size = 14)



# ============================================================
# PART E: Independent training/test datasets
# Purpose: Train on one dataset, test on a separate simulated dataset
# ============================================================

run_one_independent <- function(seed, n_train = 50, n_test = 50, p = 170, ntree = 500) {
  tryCatch({
    # -----------------------------
    # 1) Simulate TRAIN data
    # -----------------------------
    set.seed(seed)
    train <- simLong(
      n = n_train, p = p, rel_p = 6, time_points = 10,
      rho_W = 0.6, rho_Z = 0.6,
      random_sd_intercept = sqrt(0.5),
      random_sd_slope     = sqrt(3),
      noise_sd = 0.5,
      linear = TRUE
    )
    
    X_train_df <- as.data.frame(train[, -1:-5])
    
    # -----------------------------
    # 2) Fit MEGB on TRAIN
    # -----------------------------
    fit_megb <- MEGB(
      X    = as.matrix(X_train_df),
      Y    = as.matrix(train$Y),
      Z    = as.matrix(train[, 4:5]),
      id   = train$id,
      time = train$time,
      ntree = ntree,
      cv.folds = 0,
      verbose = FALSE
    )
    
    if (is.null(fit_megb$forest)) {
      return(data.frame(seed = seed, n_train = n_train, n_test = n_test,
                        rmse_megb = NA_real_, rmse_gbm = NA_real_, status = "forest_NULL"))
    }
    
    # -----------------------------
    # 3) Simulate INDEPENDENT TEST data
    # -----------------------------
    set.seed(seed + 100000)
    test <- simLong(
      n = n_test, p = p, rel_p = 6, time_points = 10,
      rho_W = 0.6, rho_Z = 0.6,
      random_sd_intercept = sqrt(0.5),
      random_sd_slope     = sqrt(3),
      noise_sd = 0.5,
      linear = TRUE
    )
    
    # make sure IDs don’t overlap train
    test$id <- test$id + max(train$id)
    
    X_test_df <- as.data.frame(test[, -1:-5])
    
    # -----------------------------
    # 4) Predict on TEST + RMSE
    # For new subjects, use fixed-only forest
    # -----------------------------
    pred_megb <- predict(fit_megb$forest, newdata = X_test_df, n.trees = ntree)
    rmse_megb <- sqrt(mean((test$Y - pred_megb)^2))
    
    # -----------------------------
    # 5) Fit GBM baseline on TRAIN, predict on TEST
    # -----------------------------
    gbm_train <- data.frame(Y = train$Y, X_train_df)
    fit_gbm <- gbm(
      Y ~ .,
      data = gbm_train,
      distribution = "gaussian",
      n.trees = ntree,
      interaction.depth = 3,
      shrinkage = 0.05,
      n.minobsinnode = 2,
      bag.fraction = 0.8,
      verbose = FALSE
    )
    
    pred_gbm <- predict(fit_gbm, newdata = X_test_df, n.trees = ntree)
    rmse_gbm <- sqrt(mean((test$Y - pred_gbm)^2))
    
    data.frame(seed = seed, n_train = n_train, n_test = n_test,
               rmse_megb = rmse_megb, rmse_gbm = rmse_gbm, status = "ok")
    
  }, error = function(e) {
    data.frame(seed = seed, n_train = n_train, n_test = n_test,
               rmse_megb = NA_real_, rmse_gbm = NA_real_,
               status = paste0("error: ", e$message))
  })
}

seeds <- 1:200
ns <- c(20, 50, 100)

res_ind <- do.call(rbind, lapply(ns, function(n0) {
  do.call(rbind, lapply(seeds, function(s) run_one_independent(seed = s, n_train = n0, n_test = n0)))
}))

table(res_ind$status)

summary_ind <- aggregate(cbind(rmse_megb, rmse_gbm) ~ n_train, data = res_ind,
                         FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                             sd = sd(x, na.rm = TRUE),
                                             n_ok = sum(!is.na(x))))
summary_ind

# difference table (MEGB - GBM)
res_ind$diff <- res_ind$rmse_megb - res_ind$rmse_gbm
diff_ind <- aggregate(diff ~ n_train, data = res_ind,
                      FUN = function(x) c(mean = mean(x, na.rm = TRUE),
                                          sd   = sd(x, na.rm = TRUE),
                                          n_ok = sum(!is.na(x))))
diff_ind



# ============================================================
# PART F: Hyperparameter tuning with 5-fold CV on training set
# Purpose: For n = 20, 50, 100:
# 1) simulate one dataset
# 2) split into training/testing
# 3) tune hyperparameters on training set only using 5-fold CV
# 4) choose lowest CV RMSE
# 5) fit final model on full training set
# 6) evaluate on held-out testing set
# ============================================================

# Make subject-level folds so each subject stays in one fold
make_subject_folds <- function(id, K = 5, seed = 123) {
  set.seed(seed)
  subj <- sample(unique(id))
  split(subj, rep(1:K, length.out = length(subj)))
}

run_tuning_pipeline <- function(n_subj, seed = 1) {
  
  set.seed(seed)
  
  # --------------------------------------------------
  # 1. Simulate one dataset
  # --------------------------------------------------
  data <- simLong(
    n = n_subj,
    p = 170,
    rel_p = 6,
    time_points = 10,
    rho_W = 0.6,
    rho_Z = 0.6,
    random_sd_intercept = sqrt(0.5),
    random_sd_slope     = sqrt(3),
    noise_sd = 0.5,
    linear = TRUE
  )
  
  # --------------------------------------------------
  # 2. Create subject-level training/testing split
  # --------------------------------------------------
  ids <- unique(data$id)
  set.seed(seed + 1000)
  test_ids <- sample(ids, size = ceiling(0.2 * length(ids)))
  
  train_data <- data[!(data$id %in% test_ids), ]
  test_data  <- data[(data$id %in% test_ids), ]
  
  folds <- make_subject_folds(train_data$id, K = 5, seed = seed)
  
  # --------------------------------------------------
  # 3. Hyperparameter grid
  # Small grid to keep runtime manageable
  # --------------------------------------------------
  grid <- expand.grid(
    ntree = c(300, 500, 800),
    shrinkage = c(0.005, 0.01, 0.02, 0.05, 0.1),
    interaction.depth = c(2, 3)
  )
  
  # --------------------------------------------------
  # 4. 5-fold CV for MEGB
  # --------------------------------------------------
  megb_cv_results <- data.frame()
  
  for (g in 1:nrow(grid)) {
    
    params <- grid[g, ]
    fold_rmse <- c()
    
    for (k in 1:5) {
      val_ids <- folds[[k]]
      
      train <- train_data[!(train_data$id %in% val_ids), ]
      val   <- train_data[(train_data$id %in% val_ids), ]
      
      X_train <- as.data.frame(train[, -1:-5])
      X_val   <- as.data.frame(val[, -1:-5])
      
      fit <- try(
        MEGB(
          X = as.matrix(X_train),
          Y = as.matrix(train$Y),
          Z = as.matrix(train[, 4:5]),
          id = train$id,
          time = train$time,
          ntree = params$ntree,
          shrinkage = params$shrinkage,
          interaction.depth = params$interaction.depth,
          n.minobsinnode = 5,
          cv.folds = 0,
          verbose = FALSE
        ),
        silent = TRUE
      )
      
      if (inherits(fit, "try-error") || is.null(fit$forest)) {
        fold_rmse <- c(fold_rmse, NA)
      } else {
        # validation subjects are unseen -> fixed-effects only
        pred <- predict(fit$forest, newdata = X_val, n.trees = params$ntree)
        rmse <- sqrt(mean((val$Y - pred)^2))
        fold_rmse <- c(fold_rmse, rmse)
      }
    }
    
    megb_cv_results <- rbind(megb_cv_results, data.frame(
      method = "MEGB",
      ntree = params$ntree,
      shrinkage = params$shrinkage,
      interaction.depth = params$interaction.depth,
      mean_rmse = mean(fold_rmse, na.rm = TRUE),
      sd_rmse = sd(fold_rmse, na.rm = TRUE),
      n_ok = sum(!is.na(fold_rmse))
    ))
  }
  
  megb_cv_results <- megb_cv_results[order(megb_cv_results$mean_rmse), ]
  best_megb <- megb_cv_results[1, ]
  
  # --------------------------------------------------
  # 5. 5-fold CV for GBM
  # --------------------------------------------------
  gbm_cv_results <- data.frame()
  
  for (g in 1:nrow(grid)) {
    
    params <- grid[g, ]
    fold_rmse <- c()
    
    for (k in 1:5) {
      val_ids <- folds[[k]]
      
      train <- train_data[!(train_data$id %in% val_ids), ]
      val   <- train_data[(train_data$id %in% val_ids), ]
      
      X_train <- as.data.frame(train[, -1:-5])
      X_val   <- as.data.frame(val[, -1:-5])
      
      gbm_train <- data.frame(Y = train$Y, X_train)
      
      fit <- try(
        gbm(
          Y ~ .,
          data = gbm_train,
          distribution = "gaussian",
          n.trees = params$ntree,
          shrinkage = params$shrinkage,
          interaction.depth = params$interaction.depth,
          n.minobsinnode = 5,
          bag.fraction = 0.8,
          verbose = FALSE
        ),
        silent = TRUE
      )
      
      if (inherits(fit, "try-error")) {
        fold_rmse <- c(fold_rmse, NA)
      } else {
        pred <- predict(fit, newdata = X_val, n.trees = params$ntree)
        rmse <- sqrt(mean((val$Y - pred)^2))
        fold_rmse <- c(fold_rmse, rmse)
      }
    }
    
    gbm_cv_results <- rbind(gbm_cv_results, data.frame(
      method = "GBM",
      ntree = params$ntree,
      shrinkage = params$shrinkage,
      interaction.depth = params$interaction.depth,
      mean_rmse = mean(fold_rmse, na.rm = TRUE),
      sd_rmse = sd(fold_rmse, na.rm = TRUE),
      n_ok = sum(!is.na(fold_rmse))
    ))
  }
  
  gbm_cv_results <- gbm_cv_results[order(gbm_cv_results$mean_rmse), ]
  best_gbm <- gbm_cv_results[1, ]
  
  # --------------------------------------------------
  # 6. Fit final MEGB on full training set with best params
  # --------------------------------------------------
  X_train_full <- as.data.frame(train_data[, -1:-5])
  X_test <- as.data.frame(test_data[, -1:-5])
  
  final_megb <- MEGB(
    X = as.matrix(X_train_full),
    Y = as.matrix(train_data$Y),
    Z = as.matrix(train_data[, 4:5]),
    id = train_data$id,
    time = train_data$time,
    ntree = best_megb$ntree,
    shrinkage = best_megb$shrinkage,
    interaction.depth = best_megb$interaction.depth,
    n.minobsinnode = 5,
    cv.folds = 0,
    verbose = FALSE
  )
  
  pred_megb <- predict(final_megb$forest, newdata = X_test, n.trees = best_megb$ntree)
  test_rmse_megb <- sqrt(mean((test_data$Y - pred_megb)^2))
  
  # --------------------------------------------------
  # 7. Fit final GBM on full training set with best params
  # --------------------------------------------------
  gbm_train_full <- data.frame(Y = train_data$Y, X_train_full)
  
  final_gbm <- gbm(
    Y ~ .,
    data = gbm_train_full,
    distribution = "gaussian",
    n.trees = best_gbm$ntree,
    shrinkage = best_gbm$shrinkage,
    interaction.depth = best_gbm$interaction.depth,
    n.minobsinnode = 5,
    bag.fraction = 0.8,
    verbose = FALSE
  )
  
  pred_gbm <- predict(final_gbm, newdata = X_test, n.trees = best_gbm$ntree)
  test_rmse_gbm <- sqrt(mean((test_data$Y - pred_gbm)^2))
  
  # --------------------------------------------------
  # 8. Return results
  # --------------------------------------------------
  list(
    n = n_subj,
    megb_cv_results = megb_cv_results,
    gbm_cv_results = gbm_cv_results,
    best_megb = best_megb,
    best_gbm = best_gbm,
    test_rmse = data.frame(
      n = n_subj,
      rmse_megb = test_rmse_megb,
      rmse_gbm = test_rmse_gbm
    )
  )
}

# --------------------------------------------------
# Run tuning pipeline for n = 20, 50, 100
# --------------------------------------------------
tune_20  <- run_tuning_pipeline(n_subj = 20, seed = 1)
tune_50  <- run_tuning_pipeline(n_subj = 50, seed = 1)
tune_100 <- run_tuning_pipeline(n_subj = 100, seed = 1)

# Best hyperparameters for each n
tune_20$best_megb
tune_20$best_gbm

tune_50$best_megb
tune_50$best_gbm

tune_100$best_megb
tune_100$best_gbm

# Final test RMSE after tuning
rbind(
  tune_20$test_rmse,
  tune_50$test_rmse,
  tune_100$test_rmse
)

# ============================================================
# Create a common theme
# ============================================================

thesis_theme <- theme_minimal(base_size = 15) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_line(color = "grey85"),
    panel.grid.major = element_line(color = "grey80")
  )

method_colors <- c("MEGB" = "#1f77b4", "GBM" = "#d62728")


# ============================================================
# FIGURE 1: Subject-level holdout
# Prediction performance with error bars
# ============================================================

subject_df <- data.frame(
  n = summary_tbl$n,
  MEGB_mean = summary_tbl$rmse_megb[, "mean"],
  MEGB_sd   = summary_tbl$rmse_megb[, "sd"],
  GBM_mean  = summary_tbl$rmse_gbm[, "mean"],
  GBM_sd    = summary_tbl$rmse_gbm[, "sd"]
)

subject_plot_df <- rbind(
  data.frame(
    n = subject_df$n,
    rmse = subject_df$MEGB_mean,
    sd = subject_df$MEGB_sd,
    method = "MEGB"
  ),
  data.frame(
    n = subject_df$n,
    rmse = subject_df$GBM_mean,
    sd = subject_df$GBM_sd,
    method = "GBM"
  )
)

subject_plot_df$method <- factor(subject_plot_df$method, levels = c("MEGB", "GBM"))

p1 <- ggplot(subject_plot_df, aes(x = n, y = rmse, color = method, group = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = rmse - sd, ymax = rmse + sd), width = 4, linewidth = 0.8) +
  scale_x_continuous(breaks = subject_df$n) +
  scale_color_manual(values = method_colors) +
  labs(
    title = "Prediction Performance Under Subject-Level Holdout",
    x = "Number of Subjects (n)",
    y = "Mean RMSE",
    color = "Method"
  ) +
  thesis_theme

print(p1)
ggsave("figure1_subject_holdout_rmse.png", p1, width = 8, height = 5.5, dpi = 300)


# ============================================================
# FIGURE 2: Time-split evaluation
# Prediction performance with error bars
# ============================================================

time_df <- data.frame(
  n = summaryB$n,
  MEGB_mean = summaryB$rmse_megb[, "mean"],
  MEGB_sd   = summaryB$rmse_megb[, "sd"],
  GBM_mean  = summaryB$rmse_gbm[, "mean"],
  GBM_sd    = summaryB$rmse_gbm[, "sd"]
)

time_plot_df <- rbind(
  data.frame(
    n = time_df$n,
    rmse = time_df$MEGB_mean,
    sd = time_df$MEGB_sd,
    method = "MEGB"
  ),
  data.frame(
    n = time_df$n,
    rmse = time_df$GBM_mean,
    sd = time_df$GBM_sd,
    method = "GBM"
  )
)

time_plot_df$method <- factor(time_plot_df$method, levels = c("MEGB", "GBM"))

p2 <- ggplot(time_plot_df, aes(x = n, y = rmse, color = method, group = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = rmse - sd, ymax = rmse + sd), width = 4, linewidth = 0.8) +
  scale_x_continuous(breaks = time_df$n) +
  scale_color_manual(values = method_colors) +
  labs(
    title = "Prediction Performance Under Time-Split Evaluation",
    x = "Number of Subjects (n)",
    y = "Mean RMSE",
    color = "Method"
  ) +
  thesis_theme

print(p2)
ggsave("figure2_time_split_rmse.png", p2, width = 8, height = 5.5, dpi = 300)


# ============================================================
# FIGURE 3: Time-split RMSE difference
# Boxplot of MEGB - GBM
# ============================================================

p3 <- ggplot(resB, aes(x = factor(n), y = diff)) +
  geom_boxplot(fill = "#6baed6", color = "black", alpha = 0.85, width = 0.65) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.9, color = "red") +
  labs(
    title = "Difference in Prediction Error Under Time-Split Evaluation",
    subtitle = "Negative values indicate improved performance of MEGB relative to GBM",
    x = "Number of Subjects (n)",
    y = "RMSE Difference (MEGB - GBM)"
  ) +
  thesis_theme

print(p3)
ggsave("figure3_time_split_difference_boxplot.png", p3, width = 8, height = 5.5, dpi = 300)


# ============================================================
# FIGURE 4: Independent dataset evaluation
# Prediction performance with error bars
# ============================================================

ind_df <- data.frame(
  n = summary_ind$n_train,
  MEGB_mean = summary_ind$rmse_megb[, "mean"],
  MEGB_sd   = summary_ind$rmse_megb[, "sd"],
  GBM_mean  = summary_ind$rmse_gbm[, "mean"],
  GBM_sd    = summary_ind$rmse_gbm[, "sd"]
)

ind_plot_df <- rbind(
  data.frame(
    n = ind_df$n,
    rmse = ind_df$MEGB_mean,
    sd = ind_df$MEGB_sd,
    method = "MEGB"
  ),
  data.frame(
    n = ind_df$n,
    rmse = ind_df$GBM_mean,
    sd = ind_df$GBM_sd,
    method = "GBM"
  )
)

ind_plot_df$method <- factor(ind_plot_df$method, levels = c("MEGB", "GBM"))

p4 <- ggplot(ind_plot_df, aes(x = n, y = rmse, color = method, group = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3.5) +
  geom_errorbar(aes(ymin = rmse - sd, ymax = rmse + sd), width = 4, linewidth = 0.8) +
  scale_x_continuous(breaks = ind_df$n) +
  scale_color_manual(values = method_colors) +
  labs(
    title = "Prediction Performance on Independent Test Data",
    x = "Number of Subjects (n)",
    y = "Mean RMSE",
    color = "Method"
  ) +
  thesis_theme

print(p4)
ggsave("figure4_independent_dataset_rmse.png", p4, width = 8, height = 5.5, dpi = 300)


# ============================================================
# FIGURE 5: Final tuned test RMSE
# Grouped bar chart
# ============================================================

tuned_test <- rbind(
  tune_20$test_rmse,
  tune_50$test_rmse,
  tune_100$test_rmse
)

tuned_plot_df <- rbind(
  data.frame(n = tuned_test$n, rmse = tuned_test$rmse_megb, method = "MEGB"),
  data.frame(n = tuned_test$n, rmse = tuned_test$rmse_gbm,  method = "GBM")
)

tuned_plot_df$method <- factor(tuned_plot_df$method, levels = c("MEGB", "GBM"))

p5 <- ggplot(tuned_plot_df, aes(x = factor(n), y = rmse, fill = method)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "black") +
  scale_fill_manual(values = method_colors) +
  labs(
    title = "Final Test RMSE After Hyperparameter Tuning",
    x = "Number of Subjects (n)",
    y = "Test RMSE",
    fill = "Method"
  ) +
  thesis_theme

print(p5)
ggsave("figure5_tuned_test_rmse.png", p5, width = 8, height = 5.5, dpi = 300)
