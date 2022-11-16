library(ggplot2)

df_m <- read.csv('C:/Users/nils/Documents/1_PhD/5_CHAPTER1/data/thermal_data/TLB_2021_deltaT.csv',
                  colClasses = c("numeric","character",rep("numeric",4)),
                  header = T,sep = ';')
df_m21 <- df_m[df_m$year == 2021,]

ggplot(df_m21, aes(y = value, x = variable,fill = variable)) +
  geom_boxplot()

ggplot(df_m21, aes(x = value, color = variable)) +
  geom_density()

pairwise.t.test(df_m21$value, df_m21$variable,
                p.adjust.method="bonferroni")


library(spatialEco)
library(landscapemetrics)
library(landscapetools)
library(terra)
library(raster)
landscape <- raster('C:/Users/nils/Documents/1_PhD/5_CHAPTER1/data/classification_data/TLB/2021/V2/TLB_2021_classified_filtered5.tif')
# landscape <- terra::rast('C:/Users/nils/Documents/1_PhD/5_CHAPTER1/data/classification_data/TLB/2021/V2/TLB_2021_classified_filtered5.tif')
landscape_s <- util_classify(landscape[100:200,200:300,1,drop = F],
                   n = 3,
                   level_names = c("dry veg.", "wet veg.","shrub veg."))

show_landscape(landscape_as_list(),
               discrete = TRUE)
# lsm_l_ta(landscape)



