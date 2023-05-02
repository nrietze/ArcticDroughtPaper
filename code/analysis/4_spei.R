# Climate data perparations for the SPEI time series
# Jakob J. Assmann jakob.assmann@uzh.ch 27 October 2022

# Dependencies
library(tidyverse)
library(ggplot2)
library(patchwork)
library(cowplot)
library(rgee)
library(sf)
library(lubridate)
library(polite)
library(SPEI)
library(rvest)

setwd('C:/data/0_Kytalyk/2_chokurdakh_meteodata')

# Webscrape climate data from http://www.pogodaiklimat.ru
# Imitate polite scraping session using bow
session <- bow("http://www.pogodaiklimat.ru/weather.php",
               force = T,
               delay =  5)

# Define year and month combinations to scrape (data available from 2011 - today) 
years_months <- expand.grid(2011:2021,1:12) %>%
  set_names("year", "month") %>%
  tibble() %>%
  arrange(year, month)

# Create output directory
dir.create("./climate/pik_ru_monthly",recursive = T)

# Scrape over year and month combinations
chokurdakh_weather <- map2(years_months$year, 
                           years_months$month,
                           function(year, month){
                             cat("Obtaining record for Year:", year, "month:", month, "\n")
                             
                             # Check weather file exists
                             if(file.exists(paste0("./climate/pik_ru_monthly/", year, "_", month, ".csv"))){
                               cat("Record exists, loading file.\n")
                               return(read_csv(paste0("./climate/pik_ru_monthly/", year, "_", month, ".csv")))
                             }
                             # Request html table for year and month
                             results <- scrape(session,
                                               query = list(id = "21946", 
                                                            bday = "1", 
                                                            fday = as.character(days_in_month(ym(paste0(year, "-", month)))), 
                                                            amonth = as.character(month),
                                                            ayear= as.character(year), 
                                                            bot = "2"))
                             # Parse table and tidy up data
                             parsed_data <- bind_cols(
                               # Side table with data and time
                               results %>% 
                                 html_nodes(".archive-table-left-column") %>% 
                                 html_table(header = T),
                               # Table with weather station observations
                               results %>% 
                                 html_nodes(".archive-table-wrap") %>% 
                                 html_table(header = T)
                             ) %>%
                               # Add leading zeros to dates and hours where they are missing
                               # Also deal with date values for October being truncated
                               mutate(
                                 `Время  (UTC),\tдата...2` = case_when(month == 10 ~ paste0(as.character(`Время  (UTC),\tдата...2`), "0"),
                                                                       TRUE ~ as.character(`Время  (UTC),\tдата...2`)),
                                 `Время  (UTC),\tдата...2` = case_when(
                                   nchar(`Время  (UTC),\tдата...2`) == 4 ~ paste0("0", `Время  (UTC),\tдата...2`),
                                   TRUE ~ paste0(`Время  (UTC),\tдата...2`)),
                                 `Время  (UTC),\tдата...1` = case_when(
                                   nchar(`Время  (UTC),\tдата...1`) == 1 ~ paste0("0", `Время  (UTC),\tдата...1`),
                                   TRUE ~ paste0(`Время  (UTC),\tдата...1`))
                               ) %>%
                               # Combine date and time, convert to POSIXct
                               mutate(date_time = 
                                        as.POSIXct(
                                          paste0(year, ".", 
                                                 `Время  (UTC),\tдата...2`,
                                                 ":", `Время  (UTC),\tдата...1`, ":00"),
                                          format = "%Y.%d.%m:%H:%M",
                                          tz = "UTC"
                                        )
                               ) %>%
                               # Select only date time, temp and precip
                               select(date_time, temp = `Т(С)`, precip = `R(мм)`)
                             
                             # Save parsed data
                             write_csv(parsed_data, paste0("./climate/pik_ru_monthly/", year, "_", month, ".csv"))
                             
                             # Return parsed data
                             return(parsed_data)
                           }) %>% 
  # combine into one dataframe
  bind_rows()

# Save to CSV file
write_csv(chokurdakh_weather,
          "./climate/pik_exports.csv")

chokurdakh_weather <- read_csv("./climate/pik_exports.csv")

# Add local time-zone variable
chokurdakh_weather$date_time_local <- with_tz(chokurdakh_weather$date_time, tzone = "Etc/GMT-11")

# Calculate daily values
chokurdakh_weather$date <- format(chokurdakh_weather$date_time_local,
                                  "%Y-%m-%d")

# Quick helper function for parsing precip values as these were
# only recorded once a day
parse_precip <- function(x){
  if(sum(is.na(x)) == length(x)) {
    return(NA)
  }
  else {
    return(sum(x, na.rm = T))
  }
}

chokurdakh_weather_daily <- chokurdakh_weather %>%
  group_by(date) %>%
  summarize(temp = mean(temp),
            precip = parse_precip(precip)) %>%
  mutate(date = as.Date(date))

# Downlad climate data for Chokurdah from GHCN-D via KNMI
# Source URLS obtained from:
# Temp: https://climexp.knmi.nl/gdcntave.cgi?id=someone@somewhere&WMO=RSM00021946&STATION=CHOKURDAH&extraargs=
# Precip: https://climexp.knmi.nl/gdcnprcp.cgi?id=someone@somewhere&WMO=RSM00021946&STATION=CHOKURDAH&extraargs=
# Snow depth: https://climexp.knmi.nl/gdcnsnwd.cgi?id=someone@somewhere&WMO=RSM00021946&STATION=CHOKURDAH&extraargs=
download.file("https://climexp.knmi.nl/data/vgdcnRSM00021946.dat",
              "./climate/vgdcnRSM00021946.dat")
download.file("https://climexp.knmi.nl/data/pgdcnRSM00021946.dat",
              "./climate/pgdcnRSM00021946.dat")
download.file("https://climexp.knmi.nl/data/dgdcnRSM00021946.dat",
              "./climate/dgdcnRSM00021946.dat")

# And also from ECA&D
# Temp: https://climexp.knmi.nl/ecatemp.cgi?id=someone@somewhere&WMO=3195&STATION=CHOKURDAH&extraargs=
# Precip: https://climexp.knmi.nl/ecaprcp.cgi?id=someone@somewhere&WMO=3195&STATION=CHOKURDAH&extraargs=
download.file("https://climexp.knmi.nl/data/teca3195.dat",
              "./climate/teca3195.dat")
download.file("https://climexp.knmi.nl/data/peca3195.dat",
              "./climate/peca3195.dat")

# Read and parse data
temp <- read_table("./climate/vgdcnRSM00021946.dat",
                   skip = 21,
                   col_names = F) %>%
  set_names(c("year", "month", "day", "temp")) %>%
  mutate(date = as.Date(paste(year, month, day), format = "%Y %m %d"))
temp_ecad <- read_table("./climate/teca3195.dat",
                        skip = 21,
                        col_names = F) %>%
  set_names(c("year", "month", "day", "temp")) %>%
  mutate(date = as.Date(paste(year, month, day), format = "%Y %m %d"))
precip <- read_table("./climate/pgdcnRSM00021946.dat",
                     skip = 22,
                     col_names = F) %>%
  set_names(c("year", "month", "day", "precip")) %>%
  mutate(date = as.Date(paste(year, month, day), format = "%Y %m %d"))
precip_ecad <- read_table("./climate/pgdcnRSM00021946.dat",
                          skip = 22,
                          col_names = F) %>%
  set_names(c("year", "month", "day", "precip")) %>%
  mutate(date = as.Date(paste(year, month, day), format = "%Y %m %d"))

snow_depth <- read_table("./climate/dgdcnRSM00021946.dat",
                         skip = 21,
                         col_names = F) %>%
  set_names(c("year", "month", "day", "snow_depth")) %>%
  mutate(date = as.Date(paste(year, month, day), format = "%Y %m %d"))

# Merge into one
climate <- full_join(temp, precip) %>%
  full_join(snow_depth) %>%
  relocate(date) 
climate_ecad <- full_join(temp_ecad, precip_ecad)

### Calculate the SPEI

# combine source datasets and calculate monthly values
spei <- bind_rows(climate_ecad %>%
                    mutate(year = as.numeric(format(date, "%Y")),
                           month = as.numeric(format(date, "%m"))) %>%
                    filter(!(year %in% c(2014, 2016, 2017, 2019))) %>%
                    group_by(year, month) %>%
                    summarize(temp = mean(temp, na.rm = T),
                              precip = sum(precip, na.rm = T)) %>%
                    mutate(water_bal = precip - as.numeric(thornthwaite(temp, 70.62))),
                  chokurdakh_weather_daily %>%
                    mutate(year = as.numeric(format(date, "%Y")),
                           month = as.numeric(format(date, "%m"))) %>%
                    filter(year %in%  c(2014, 2016, 2017, 2019:2021)) %>%
                    group_by(year, month) %>%
                    summarize(temp = mean(temp, na.rm = T),
                              precip = sum(precip, na.rm = T)) %>%
                    mutate(water_bal = precip - as.numeric(thornthwaite(temp, 70.62)))
) %>%
  ungroup() %>%
  arrange(year, month) %>%
  mutate(spei_3_months = as.numeric(spei(water_bal, 3)$fitted),
         spei_6_months = as.numeric(spei(water_bal, 6)$fitted),
         spei_9_months = as.numeric(spei(water_bal, 9)$fitted),
         spei_12_months = as.numeric(spei(water_bal, 12)$fitted),
         spei_24_months = as.numeric(spei(water_bal, 24)$fitted)) %>%
  filter(year != 1944)

write_csv(spei, "./climate/spei_monthly.csv")