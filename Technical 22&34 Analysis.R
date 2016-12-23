library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)

# Factor technical_22 Correlation
corrplot.mixed(
  DF %>% group_by(Factor = technical_22, timestamp) %>% summarise(y = mean(y)) %>% spread(Factor, y) %>% .[, -1] %>% cor(),
  title = "Technical 22"
)

# Factor technical_34 Correlation
corrplot.mixed(
  DF %>% group_by(Factor = technical_34, timestamp) %>% summarise(y = mean(y)) %>% spread(Factor, y) %>% .[, -1] %>% cor(),
  title = "Technical 34"
)

# Combined Factor Correlation
corrplot.mixed(
  DF %>% group_by(Factor = paste0(technical_22, "\n", technical_34), timestamp) %>% summarise(y = mean(y)) %>% spread(Factor, y) %>% .[, -1] %>% cor(),
  title = "Combined 34"
)

# Factor technical_22 Equity Curve
print(
  DF %>% group_by(Factor = technical_22, timestamp) %>% summarise(y = mean(y)) %>% group_by(Factor) %>% mutate(Cumm = cumsum(y)) %>% 
    ggplot(aes(x = timestamp, y = Cumm, colour = factor(Factor))) + geom_line() + ggtitle("Technical 22")
)

# Factor technical_34 Equity Curve
print(
  DF %>% group_by(Factor = technical_34, timestamp) %>% summarise(y = mean(y)) %>% group_by(Factor) %>% mutate(Cumm = cumsum(y)) %>% 
    ggplot(aes(x = timestamp, y = Cumm, colour = factor(Factor))) + geom_line() + ggtitle("Technical 34")
)

# Combined Factor Equity Curve
print(
  DF %>% group_by(Factor = paste0(technical_22, " & ", technical_34), timestamp) %>% summarise(y = mean(y)) %>% group_by(Factor) %>% mutate(Cumm = cumsum(y)) %>% 
    ggplot(aes(x = timestamp, y = Cumm, colour = factor(Factor))) + geom_line() + ggtitle("Combined")
)


