# R container for the Titanic logistic regression model
# rocker/r-ver pins both the R version and the CRAN snapshot date
FROM rocker/r-ver:4.4.1

ENV DATA_DIR=/app/data

WORKDIR /app

# Install packages first (cached layer)
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the script
COPY titanic.R .

# The data is NOT baked into the image; it is mounted at /app/data at run time
CMD ["Rscript", "titanic.R"]
