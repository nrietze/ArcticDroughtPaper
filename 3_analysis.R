library(ggplot2)
library(GGally)
library(lubridate)
library(scales)
library(zoo)
library(latex2exp)

myprettyoptions = list(theme_bw(),
                       theme(panel.grid.major = element_line(colour = "lightgrey"),
                             plot.title = element_text(size=16, hjust = 0.5),
                             axis.title.x = element_text(size=16), axis.text.x  = element_text(size=12),
                             axis.title.y = element_text(size=16), axis.text.y  = element_text(size=12),
                             strip.text.x = element_text(size = 14), strip.text.y = element_text(size = 14),
                             legend.text = element_text(size=12), legend.title = element_text(size=14), title = element_text(size=16)))

HOMEDIR <- r'(\\ieu-fs.uzh.ch\Groups\Schaepman\Gr_Schaepman\Siberia_drought_precipitation\Kytalyk_Data_ALL\TRain_Data\HOBO)'

df <- read.csv2(paste(HOMEDIR,'/merged_data' , 'data_HOBO.csv',sep = '/'))
df$datetime <- as.POSIXct(strptime(df$local_time,format = '%Y-%m-%dT%H:%M:%SZ',tz="Asia/Srednekolymsk"))

df_fluxtower <- read.delim(file = 'C:/data/0_Kytalyk/7_flux_tower/CR3000_Kytalyk_Meteo.dat',
                           skip = 1,sep = '\t',
                           na.strings = 'NAN')
df_fluxtower <- df_fluxtower[3:nrow(df_fluxtower),]
df_fluxtower[,2:ncol(df_fluxtower)] <- lapply(df_fluxtower[,2:ncol(df_fluxtower)],as.numeric)
df_fluxtower$datetime <- as.POSIXct(strptime(df_fluxtower$TIMESTAMP,format = '%m/%d/%y %H:%M',tz="Asia/Srednekolymsk"))

SelectData <- function(df, s, tr,period){
  df_filtered <- df %>% 
    filter(datetime >= '2020-06-01' & site == s & treatment == tr) %>% 
    filter(sm_depth_1 >= 0 & sm_depth_2 >= 0) %>% 
    mutate(month = months(datetime,abbreviate = TRUE)) %>%
    filter(month %in% c("Jun","Jul", "Aug"))
  
  df_period <- na.omit(df_filtered) %>% 
    mutate(year_period = floor_date(datetime,period)) %>% 
    group_by(year_period) %>% 
    summarize(soil_temperature = mean(temperature),
              sm_depth_1 = mean(sm_depth_1),
              sm_depth_2 = mean(sm_depth_2))
  
  df_meteo <- read.csv2(paste(HOMEDIR,'/merged_data' , sprintf( 'data_HOBO_meteo_%s.csv',s),sep = '/'))
  df_meteo$datetime <- as.POSIXct(strptime(df_meteo$local_time,format = '%m.%d.%y %H:%M:%S %p',tz="Asia/Srednekolymsk"))
  
  df_meteo_period <- df_meteo %>% 
    filter(datetime >= '2020-06-01') %>% 
    mutate(year_period = floor_date(datetime,period)) %>% 
    group_by(year_period) %>% 
    summarize(across(seq(3,8), mean,
                     na.rm = TRUE)) %>% 
    mutate(deltaT_HOBO = .[[2]] - .[[4]])
  
  df_fluxtower_period <- df_fluxtower %>% 
    filter(datetime >= '2020-06-01') %>% 
    mutate(year_period = floor_date(datetime,period)) %>% 
    group_by(year_period) %>% 
    summarize(T_air = mean(Barani_Temp_2_Avg, na.rm = TRUE))
  
  df_period <- list(df_period,df_meteo_period,df_fluxtower_period) %>% reduce(left_join, by = "year_period")
  
  return(list(df_filtered, df_period))
}

site <- 'lakebed'
treatment <- 'control'
period <- 'day'

dfl <- SelectData(df, site, treatment,period)
df_filtered <- dfl[[1]]
df_period <- dfl[[2]]

df_period <- df_period %>% 
  mutate(deltaT_fluxtower = .[[5]] - T_air)

summary(df_period)
ggpairs(df_period)

ggplot(data = df_period) +
  geom_point(aes(x = yday(datetime) , y = sm_depth_1,color = year(datetime)), size = 1) +
  theme_bw() + ggtitle("Soil moisture, 1 cm depth in mineral soil") + 
  xlab('Day of year') + ylab("volumetric water content, m3/m3") +
  myprettyoptions

ggplot(data = df_period) +
  geom_point(aes(x = yday(datetime) , y = sm_depth_2,color = year(datetime)), size = 1) +
  theme_bw() + ggtitle("Soil moisture, 10 cm depth in mineral soil") + 
  xlab('Day of year') + ylab("volumetric water content, m3/m3") +
  myprettyoptions

df_sd <- aggregate(deltaT_fluxtower ~ year(year_period) + week(year_period),sd,data = df_period)
df_sdd <- df_period %>% 
  mutate(year_period = floor_date(datetime,'2 days')) %>% 
  group_by(year_period) %>% 
  summarize(sigma_deltaT_HOBO = sd(deltaT_HOBO ),
            sigma_deltaT_FT = sd(deltaT_fluxtower ),
            mu_sm_depth_1 = mean(sm_depth_1),
            mu_sm_depth_2 = mean(sm_depth_2))
head(df_sdd)

ggplot(data = df_sdd) +
  geom_point(aes(x = mu_sm_depth_1 ,y = sigma_deltaT_HOBO, color = year(year_period)), size = 1) +
  theme_bw() + 
  xlab("Mean soil moisture at 1 cm") + ylab(TeX("Std. deviation of $T_{15cm}$ - $T_{60cm}$")) +
  myprettyoptions

ggplot(data = df_sdd) +
  geom_point(aes(x = yday(year_period) , y = sigma_deltaT_HOBO,color = year(year_period)), size = 1) +
  theme_bw() + 
  xlab('Day of year') + ylab("std. deviation of deltaT") +
  myprettyoptions

ggplot(data = df_period) +
  geom_point(aes(x = deltaT_HOBO , y = deltaT_fluxtower,color = datetime), size = 1) +
  theme_bw() + ggtitle("Temperature differences (lower - upper (°C)") + 
  xlab(TeX("$T_{15cm}$ - $T_{60cm}$")) +
  ylab(TeX("$T_{15cm}$ - $T_{2m}$")) +
  myprettyoptions


mod1 <- lm(deltaT_fluxtower ~ deltaT_HOBO,data = df_period)
summary(mod1)

library(lme4)
mod2 <- lmer(deltaT_fluxtower ~ deltaT_HOBO + (1|year),data = df_period)
summary(mod2)
