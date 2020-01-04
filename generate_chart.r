library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)

data <- read.csv('C:/Users/Mateusz/source/repos/SN_Project3/results.csv')



data_first_fixed <- data %>%
  filter(stringr::str_detect(architecture, '30-')) %>%
  distinct(train, architecture, inteprolate, applyWindTransformation, .keep_all = TRUE) %>%
  separate(architecture, c("first_layer", "second_layer"), sep = "-") %>%
  mutate(first_layer = as.numeric(substr(first_layer,2,nchar(first_layer)))) %>%
  mutate(second_layer = as.numeric(substr(second_layer,2,nchar(second_layer)-1))) %>%
  arrange(first_layer, second_layer)

data_second_fixed <- data %>%
  filter(stringr::str_detect(architecture, '.*10')) %>%
  distinct(train, architecture,  inteprolate, applyWindTransformation, .keep_all = TRUE) %>%
  separate(architecture, c("first_layer", "second_layer"), sep = "-") %>%
  mutate(first_layer = as.numeric(substr(first_layer,2,nchar(first_layer)))) %>%
  mutate(second_layer = as.numeric(substr(second_layer,2,nchar(second_layer)-1))) %>%
  arrange(first_layer, second_layer)


dose.labs <- c("Without wind transformation", "With wind transformation")
names(dose.labs) <- c("0", "1")

supp.labs <- c("Without interpolation", "With interpolation")
names(supp.labs) <- c("0", "1")


p1 <- ggplot(aes(y = temperatureAvgError, group=train, color=train, x = reorder(paste('(', first_layer,',',second_layer,')'), second_layer)), data = data_first_fixed) + 
  geom_errorbar(aes(ymin=max(0, temperatureAvgError-temperatureAvgStd), ymax=temperatureAvgError+temperatureAvgStd), width=.1) +
  geom_line(linetype="dashed") +
  geom_point()+
  scale_linetype_manual(values=c("twodash", "dotted"))+
  theme(text=element_text(family="Tahoma")) +
  facet_grid(applyWindTransformation~inteprolate, labeller=labeller(applyWindTransformation = dose.labs, inteprolate = supp.labs))

p1 + labs(title = "Results of Neural Network", x = "Architecture", y = "Average error", caption = "2019")

ggsave("XD.png")

