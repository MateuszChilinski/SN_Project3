library(plyr)
library(dplyr)
library(ggplot2)

data <- read.csv('results.csv')

data2 <- data %>%
  filter(inteprolate == 0 & applyWindTransformation == 0)

p1 <- ggplot() + 
  geom_line(aes(y = temperatureAvgError, group=train, color=train, x = architecture), data = data2) +
  theme(text=element_text(family="Tahoma"))

p1 + labs(title = "Results of Neural Network", x = "Architecture", y = "Average error", caption = "2019")

