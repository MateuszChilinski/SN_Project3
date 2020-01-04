library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
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
  geom_errorbar(aes(ymin=max(0, temperatureAvgError-temperatureAvgStd), ymax=temperatureAvgError+temperatureAvgStd), width=.1, position=position_dodge(width=0.2)) +
  geom_line(linetype="dashed", position=position_dodge(width=0.2)) +
  geom_point(position=position_dodge(width=0.2))+
  scale_linetype_manual(values=c("twodash", "dotted"))+
  facet_grid(applyWindTransformation~inteprolate, labeller=labeller(applyWindTransformation = dose.labs, inteprolate = supp.labs))

p1 + labs(color = "Sets", title = "Results of Neural Network", x = "Architecture", y = "Average temp. error [°C]", caption = "Mateusz Chilinski, Bartlomiej Chechlinski, 2020")

ggsave("1.png", width = 10, height = 6, dpi = 300)

# 2nd

p2 <- ggplot(aes(y = temperatureAvgError, group=train, color=train, x = reorder(paste('(', first_layer,',',second_layer,')'), first_layer)), data = data_second_fixed) + 
  geom_errorbar(aes(ymin=max(0, temperatureAvgError-temperatureAvgStd), ymax=temperatureAvgError+temperatureAvgStd), width=.1, position=position_dodge(width=0.2)) +
  geom_line(linetype="dashed", position=position_dodge(width=0.2)) +
  geom_point(position=position_dodge(width=0.2))+
  scale_linetype_manual(values=c("twodash", "dotted"))+
  facet_grid(applyWindTransformation~inteprolate, labeller=labeller(applyWindTransformation = dose.labs, inteprolate = supp.labs))

p2 + labs(color = "Sets", title = "Results of Neural Network", x = "Architecture", y = "Average temp. error [°C]", caption = "Mateusz Chilinski, Bartlomiej Chechlinski, 2020")

ggsave("2.png", width = 10, height = 6, dpi = 300)

p3 <- ggplot(aes(y = windGoodPredictions, group=train, color=train, fill=train, x = reorder(paste('(', first_layer,',',second_layer,')'), second_layer)), data = data_first_fixed) + 
  geom_col(position=position_dodge(width=1)) +
  scale_y_continuous(limits=c(85,100),oob = rescale_none) +
  facet_grid(applyWindTransformation~inteprolate, labeller=labeller(applyWindTransformation = dose.labs, inteprolate = supp.labs))

p3 + labs(fill="Sets", color = "Sets", title = "Results of Neural Network", x = "Architecture", y = "Wind correctly predicted [%]", caption = "Mateusz Chilinski, Bartlomiej Chechlinski, 2020")

ggsave("3.png", width = 10, height = 6, dpi = 300)

p4 <- ggplot(aes(y = windGoodPredictions, group=train, color=train, fill=train, x = reorder(paste('(', first_layer,',',second_layer,')'), first_layer)), data = data_second_fixed) + 
  geom_col(position=position_dodge(width=1)) +
  scale_y_continuous(limits=c(85,100),oob = rescale_none) +
  facet_grid(applyWindTransformation~inteprolate, labeller=labeller(applyWindTransformation = dose.labs, inteprolate = supp.labs))

p4 + labs(fill="Sets", color = "Sets", title = "Results of Neural Network", x = "Architecture", y = "Wind correctly predicted [%]", caption = "Mateusz Chilinski, Bartlomiej Chechlinski, 2020")

ggsave("4.png", width = 10, height = 6, dpi = 300)