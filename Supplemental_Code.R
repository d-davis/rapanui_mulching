library(RStoolbox)
library(caret)
library(MIAmaxent)
library(randomForest)
library(e1071)
library(raster)
library(spatstat)
library(rgdal)
library(sf)
library(stars)
library(rgeos)
library(dplyr)
library(maptools)
library(sp)

#set working directory
setwd("C:/Users/dylan/Downloads/Mosaic_PS_16bit")


#Open VNIR files
B <- raster::raster("MS_8b_rs.tif", band = 1)
G <- raster::raster("MS_8b_rs.tif", band = 2)
Y <- raster::raster("MS_8b_rs.tif", band = 3)
R <- raster::raster("MS_8b_rs.tif", band = 4)
RE <- raster::raster("MS_8b_rs.tif", band = 5)
NIR1 <- raster::raster("MS_8b_rs.tif", band = 6)
NIR2 <- raster::raster("MS_8b_rs.tif", band = 7)

VNIR_stk <- raster::stack(B, G, Y, R, RE, NIR1, NIR2)

#Set working directory
setwd("C:/Users/dylan/Documents/School_Work/Rapa_Nui")

#Load image files
#SWIR
B1 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 1)
B2 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 2)
B3 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 3)
B4 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 4)
B5 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 5)
B6 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 6)
B7 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 7)
B8 <- raster::raster("Delivery/Mosaic_8bit/Easter_Island_SWIR_8Band_8bit.tif", band = 8)

SWIR_stk <- raster::stack(B1, B2, B3, B4, B5, B6, B7, B8)

#load AOI shapefile
AOI <- rgdal::readOGR("AOI.shp") 

#clip and merge SWIR and VNIR files

SWIR_clp <- raster::crop(SWIR_stk, AOI)

VNIR_clp <- raster::crop(VNIR_stk, AOI)

#Pansharpen SWIR using VNIR
WV3_PS <- RStoolbox::panSharpen(SWIR_stk, NIR1, method = "pca")

#writeRaster(WV3_PS, "SWIR_PS.tif") #Save pansharpened image as tif file

#Load pansharpened SWIR image

PS_1 <- raster::raster("SWIR_PS.tif", band = 1)
PS_2 <- raster::raster("SWIR_PS.tif", band = 2)
PS_3 <- raster::raster("SWIR_PS.tif", band = 3)
PS_4 <- raster::raster("SWIR_PS.tif", band = 4)
PS_5 <- raster::raster("SWIR_PS.tif", band = 5)
PS_6 <- raster::raster("SWIR_PS.tif", band = 6)
PS_7 <- raster::raster("SWIR_PS.tif", band = 7)
PS_8 <- raster::raster("SWIR_PS.tif", band = 8)

WV3_PS <-raster::stack(PS_1, PS_2, PS_3, PS_4, PS_5, PS_6, PS_7, PS_8)
#Merge VNIR and SWIR

WV3 <- raster::stack(WV3_PS, VNIR_stk)
#Load training data
train <- rgdal::readOGR("Training_v7.shp") 

##Plot Training Data
olpar <- par(no.readonly = TRUE) # back-up par
colors <- c("yellow", "red", "blue", 'green', 'black')
plot(B1)
plot(train, col = colors[train$Class], pch = 19)

##CLASSIFY SWIR DATASET

## Fit classifier (splitting training into 80\% training data, 20\% validation data)

MaxEnt_SWIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                      algorithm = "maxent", tuneLength = 1, trainPartition = 0.8, filename = "MxEnt_v3.tif") 

MaxEnt_SWIR #Display model results, accuracy metrics, and confusion matrix

MLC_SWIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                      model = "mlc", tuneLength = 1, trainPartition = 0.8, filename = "MLC_v3.tif") 

MLC_SWIR #Display model results, accuracy metrics, and confusion matrix

RF_SWIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                   model = "rf", tuneLength = 1, trainPartition = 0.8, filename = "RF_v3.tif") 

RF_SWIR #Display model results, accuracy metrics, and confusion matrix


##CLASSIFY VNIR DATASET

## Fit classifier (splitting training into 80\% training data, 20\% validation data)

MaxEnt_VNIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                           algorithm = "maxent", tuneLength = 1, trainPartition = 0.8, filename = "MxEnt_v3.tif") 

MaxEnt_VNIR #Display model results, accuracy metrics, and confusion matrix

MLC_VNIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                        model = "mlc", tuneLength = 1, trainPartition = 0.8, filename = "MLC_v3.tif") 

MLC_VNIR #Display model results, accuracy metrics, and confusion matrix

RF_VNIR       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                       model = "rf", tuneLength = 1, trainPartition = 0.8, filename = "RF_v3.tif") 

RF_VNIR #Display model results, accuracy metrics, and confusion matrix

##CLASSIFY Pansharpened SWIR DATASET

## Fit classifier (splitting training into 80\% training data, 20\% validation data)

MaxEnt_PS       <- RStoolbox::superClass(SWIR_stk, trainData = train, responseCol = "Name", 
                                           algorithm = "maxent", tuneLength = 1, trainPartition = 0.8, filename = "MxEnt_v3.tif") 

MaxEnt_PS #Display model results, accuracy metrics, and confusion matrix

