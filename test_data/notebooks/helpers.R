library(ggplot2)
library(dplyr)

load_results <- function(path) {
  data <- read.csv(path)
  return(data)
}

plot_training_curve <- function(data) {
  ggplot(data, aes(x=epoch, y=loss)) + geom_line() + theme_minimal()
}
