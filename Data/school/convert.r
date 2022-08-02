setwd("/Users/ziping.xu/Desktop/DeepWeightDecay/data/school")
library(tidyverse)
library(dplyr)

dat = read.csv("data.csv")
dat$student = seq(1, dim(dat)[1])

list_to_convert = c("YEAR", "SCHGEN", "SCHDEN", "SEX", "ETHN", "VRBAND")

for(name in list_to_convert){
  uni = unique(dat[[name]])
  dat[[name]] = factor(dat[[name]])
  levels(dat[[name]]) = paste(name, uni, sep='_')
  dat = dat %>% mutate(dummy=1) %>%
        spread(key=name,value=dummy, fill=0)
}
head(dat)
dat$student = NULL

write.csv(dat, file = "data_dummpy.csv")
dat %>% count(SCHOOL, sort = TRUE)
