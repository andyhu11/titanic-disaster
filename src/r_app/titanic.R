# Titanic survival prediction with logistic regression (R).
#
# Reads train.csv / test.csv from DATA_DIR (default /app/data, mounted by Docker),
# prints every data change, fits a logistic regression (glm), reports accuracy,
# and writes predictions for test.csv.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

data_dir <- Sys.getenv("DATA_DIR", "/app/data")
set.seed(42)

# R model uses a slightly different feature set than the Python model:
# SibSp/Parch are summarised by FamilySize (keeping all three is perfectly collinear in glm),
# and Title is used only for Age imputation.
model_formula <- Survived ~ Pclass + Sex + Age + Fare + Embarked +
  FamilySize + IsAlone + HasCabin

section <- function(title) {
  line <- strrep("=", 70)
  cat("\n", line, "\n", title, "\n", line, "\n", sep = "")
}

load_csv <- function(name) {
  path <- file.path(data_dir, name)
  if (!file.exists(path)) {
    cat(sprintf(paste0(
      "[ERROR] %s not found.\nDownload train.csv and test.csv from ",
      "https://www.kaggle.com/competitions/titanic/data into src/data/ ",
      "and mount that folder with -v (see README).\n"), path))
    quit(status = 1)
  }
  df <- as.data.frame(read_csv(path, show_col_types = FALSE))
  cat(sprintf("Loaded %s: %d rows x %d columns\n", name, nrow(df), ncol(df)))
  df
}

pct <- function(x) sprintf("%.1f%%", 100 * x)

fmt_counts <- function(x) {
  tab <- table(x)
  paste(names(tab), as.integer(tab), sep = "=", collapse = ", ")
}

extract_title <- function(names) {
  t <- trimws(sub("^[^,]*,\\s*([^.]+)\\..*$", "\\1", names))
  t[t %in% c("Mlle", "Ms")] <- "Miss"
  t[t == "Mme"] <- "Mrs"
  ifelse(t %in% c("Mr", "Mrs", "Miss", "Master"), t, "Rare")
}

# Learn every fill value and factor level from the TRAINING data only
fit_stats <- function(train) {
  titles <- extract_title(train$Name)
  list(
    age_by_title  = tapply(train$Age, titles, median, na.rm = TRUE),
    age_overall   = median(train$Age, na.rm = TRUE),
    fare_by_class = tapply(train$Fare, train$Pclass, median, na.rm = TRUE),
    embarked_mode = names(which.max(table(train$Embarked))),
    sex_levels    = sort(unique(train$Sex)),
    emb_levels    = sort(unique(na.omit(train$Embarked)))
  )
}

# Apply the same cleaning steps to any dataset, printing each change
preprocess <- function(df, stats, label) {
  cat(sprintf("\n--- Cleaning %s ---\n", label))

  # 1. Title from Name
  df$Title <- extract_title(df$Name)
  cat(sprintf("[ADD] Title extracted from Name -> counts: %s\n", fmt_counts(df$Title)))

  # 2. Age: fill by median age of the passenger's title
  miss <- is.na(df$Age)
  fill <- unname(stats$age_by_title[df$Title])
  fill[is.na(fill)] <- stats$age_overall
  df$Age[miss] <- fill[miss]
  cat(sprintf("[FILL] Age: %d missing values filled with the median age per Title (%s)\n",
              sum(miss),
              paste(names(stats$age_by_title), round(stats$age_by_title, 1),
                    sep = "=", collapse = ", ")))

  # 3. Fare: fill by median fare of the passenger class
  miss <- is.na(df$Fare)
  df$Fare[miss] <- unname(stats$fare_by_class[as.character(df$Pclass[miss])])
  cat(sprintf("[FILL] Fare: %d missing values filled with the median Fare per Pclass\n",
              sum(miss)))

  # 4. Embarked: fill with the most common port
  miss <- is.na(df$Embarked)
  df$Embarked[miss] <- stats$embarked_mode
  cat(sprintf("[FILL] Embarked: %d missing values filled with mode '%s'\n",
              sum(miss), stats$embarked_mode))

  # 5-6. Family and cabin features
  df <- df %>%
    mutate(FamilySize = SibSp + Parch + 1,
           IsAlone    = as.integer(FamilySize == 1),
           HasCabin   = as.integer(!is.na(Cabin)))
  cat(sprintf(paste0("[ADD] FamilySize = SibSp + Parch + 1 (mean %.2f); ",
                     "IsAlone = 1 if FamilySize == 1 (%d passengers alone)\n"),
              mean(df$FamilySize), sum(df$IsAlone)))
  cat(sprintf("[ADD] HasCabin = 1 if a cabin is recorded (%d passengers)\n",
              sum(df$HasCabin)))

  # 7. Categorical columns -> factors with the training levels
  df$Sex      <- factor(df$Sex, levels = stats$sex_levels)
  df$Embarked <- factor(df$Embarked, levels = stats$emb_levels)
  cat(sprintf("[ENCODE] Sex -> factor (%s); Embarked -> factor (%s)\n",
              paste(stats$sex_levels, collapse = "/"),
              paste(stats$emb_levels, collapse = "/")))

  # 8. Columns not used by the model
  cat("[DROP] Not used as features: PassengerId, Name, Ticket, Cabin, SibSp, Parch, Title\n")

  used <- all.vars(model_formula)[-1]
  cat(sprintf("[CHECK] Missing values left in features: %d\n", sum(is.na(df[, used]))))
  df
}

accuracy <- function(fit, data) {
  pred <- as.integer(predict(fit, newdata = data, type = "response") > 0.5)
  mean(pred == data$Survived)
}

# ---------------------------------------------------------------------- #
section("STEP 0 - Environment check")
cat(sprintf("%s | readr %s | dplyr %s\n", R.version.string,
            as.character(packageVersion("readr")), as.character(packageVersion("dplyr"))))
cat(sprintf("Reading data from: %s\n", data_dir))

# ---------------------------------------------------------------------- #
section("STEP 1 - Load train.csv")
train <- load_csv("train.csv")
cat("\nFirst 5 rows:\n")
print(head(train, 5))
cat("\nMissing values per column:\n")
na_counts <- colSums(is.na(train))
print(na_counts[na_counts > 0])

# ---------------------------------------------------------------------- #
section("STEP 2 - Explore: a few interesting notes")
cat(sprintf("Overall survival rate: %s\n", pct(mean(train$Survived))))
cat("\nSurvival rate by Sex:\n")
train %>% group_by(Sex) %>% summarise(rate = pct(mean(Survived)), .groups = "drop") %>%
  as.data.frame() %>% print(row.names = FALSE)
cat("\nSurvival rate by Pclass:\n")
train %>% group_by(Pclass) %>% summarise(rate = pct(mean(Survived)), .groups = "drop") %>%
  as.data.frame() %>% print(row.names = FALSE)
kids <- !is.na(train$Age) & train$Age < 16
adults <- !is.na(train$Age) & train$Age >= 16
cat(sprintf("\nSurvival rate for children (<16): %s vs adults: %s\n",
            pct(mean(train$Survived[kids])), pct(mean(train$Survived[adults]))))

# ---------------------------------------------------------------------- #
section("STEP 3 - Adjust the data (training set)")
stats <- fit_stats(train)
train_clean <- preprocess(train, stats, "train.csv")

# ---------------------------------------------------------------------- #
section("STEP 4 - Logistic regression on the training set")
idx_by_class <- split(seq_len(nrow(train_clean)), train_clean$Survived)
val_idx <- unlist(lapply(idx_by_class, function(i) sample(i, round(0.2 * length(i)))))
tr  <- train_clean[-val_idx, ]
val <- train_clean[val_idx, ]
cat(sprintf("Split train.csv 80/20 (stratified): %d training rows, %d validation rows\n",
            nrow(tr), nrow(val)))
fit <- glm(model_formula, data = tr, family = binomial)
cat("Model: glm(family = binomial)\nFormula: ")
print(model_formula, showEnv = FALSE)
cat("\nCoefficients (log-odds; + raises, - lowers survival odds):\n")
print(round(coef(fit), 3))

# ---------------------------------------------------------------------- #
section("STEP 5 - Accuracy on the training set")
train_acc <- accuracy(fit, tr)
val_acc   <- accuracy(fit, val)
cat(sprintf("Training accuracy   (80%% of train.csv): %.4f\n", train_acc))
cat(sprintf("Validation accuracy (20%% held out):     %.4f\n", val_acc))

fit <- glm(model_formula, data = train_clean, family = binomial)
cat(sprintf("Refit on 100%% of train.csv -> training accuracy: %.4f\n",
            accuracy(fit, train_clean)))

# ---------------------------------------------------------------------- #
section("STEP 6 - Load test.csv and predict")
test <- load_csv("test.csv")
test_clean <- preprocess(test, stats, "test.csv (using values learned from train.csv)")
test_pred <- as.integer(predict(fit, newdata = test_clean, type = "response") > 0.5)
cat(sprintf("\nPredicted survivors in test.csv: %d of %d (%s)\n",
            sum(test_pred), length(test_pred), pct(mean(test_pred))))

out_path <- file.path(data_dir, "predictions_r.csv")
write_csv(data.frame(PassengerId = test$PassengerId, Survived = test_pred), out_path)
cat(sprintf("Saved predictions to %s (Kaggle submission format)\n", out_path))

# ---------------------------------------------------------------------- #
section("STEP 7 - Accuracy on the test set")
if ("Survived" %in% names(test)) {
  cat(sprintf("Test accuracy: %.4f\n", mean(test_pred == test$Survived)))
} else {
  cat("test.csv has NO 'Survived' column - Kaggle keeps the true labels hidden,\n")
  cat("so test accuracy cannot be computed locally.\n")
  cat(sprintf("Best local estimate (held-out validation accuracy): %.4f\n", val_acc))
  cat("To get the real test score, submit predictions_r.csv on Kaggle.\n")
}

section("DONE")
