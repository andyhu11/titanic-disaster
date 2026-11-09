# Install the R packages needed by titanic.R.
# The rocker/r-ver:4.4.1 base image points CRAN at a dated snapshot,
# so these installs always resolve to the same package versions.

pkgs <- c("readr", "dplyr")

install.packages(pkgs)

# install.packages() only warns on failure - fail the Docker build instead
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  stop("Failed to install: ", paste(missing, collapse = ", "))
}

cat("\nInstalled package versions:\n")
for (p in pkgs) cat(sprintf("  %-6s %s\n", p, as.character(packageVersion(p))))
