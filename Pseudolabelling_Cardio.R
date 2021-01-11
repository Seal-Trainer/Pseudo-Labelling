
## Instalacion de los paquetes
# install.packages("e1071")
# install.packages("glmnet")
# install.packages("magrittr")
# install.packages("dplyr")
# install.packages("imager")
# install.packages("caret")
# install.packages("confusionMatrix")
# install.packages("mltools")

## Cargamos las librerias necesarias
library(dplyr)
library(glmnet)
library(magrittr)
library(imager)
library(caret)
library(e1071)
library(mltools)
library(tidyr)

setwd("~/Documents/Thesis/Code")

source("Metrics.R")

dataset <- read.csv(file = "Data/cardio_train.csv", sep = ";")

## Lo primero que haremos sera
## poner la variable respuesta en terminos
## de 0 y 1.

temp <- dataset

temp$id <- NULL

temp$age <- (temp$age / 365)
temp$age <- round(temp$age)
temp$cardio <- factor(temp$cardio, labels = c(0,1))

predictor <- temp[ , 1:11]
predictor <- as.matrix(predictor)

variableRespuesta <- temp[ , 12]
variableRespuestaFES <- temp[ , 12]

nIteracciones <- 20
## Cantidad de filas
nFilas <- nrow(predictor)

resultados <- data.frame(Particion = numeric(),
                         Posicion = numeric(), 
                         Observado = character(),
                         Predicciones = character(),
                         PrediccionesFE = character(),
                         PrediccionesFES = character())

for(i in 1:nIteracciones)
{
  # i = 1
  print(i)
  ## Falso Etiquedado = FE
  indicesEntrenamiento <- sample(nFilas, nFilas/70)
  indicesTesteo <- setdiff(1:nFilas, indicesEntrenamiento)
  indicesTesteo <- sample(indicesTesteo, nFilas/140)
  indicesFE <- setdiff(1:nFilas, c(indicesEntrenamiento, 
                                   indicesTesteo))
  indicesFE <- sample(indicesFE, nFilas/140)                                              
  
  indicesUnion <- c(indicesEntrenamiento, indicesFE)
  
  ### Falsas etiquetas
  modeloFE <- cv.glmnet(predictor[indicesEntrenamiento, ], 
                        variableRespuesta[indicesEntrenamiento],
                        family = "binomial")
  
  FE <- data.frame(clase = c(predict(modeloFE, predictor[indicesFE, ], 
                                     type = "class")),
                   prob =  c(predict(modeloFE, predictor[indicesFE, ], 
                                     type = "response")),
                   indice = indicesFE)
  
  indicesFES <- which(FE$prob >= 0.75 | FE$prob <= 0.25)
  FES <- FE[indicesFES, ]
  
  variableRespuesta[FE$indice] <- FE$clase
  variableRespuestaFES[FES$indice] <- FES$clase
  
  indicesUnionFES <- c(indicesEntrenamiento, FES$indice)
  
  ### Fase de testeo
  modeloFETesteo <- cv.glmnet(predictor[indicesUnion, ], 
                              variableRespuesta[indicesUnion],
                              family = "binomial")
  
  modeloFESTesteo <- cv.glmnet(predictor[indicesUnionFES,],
                               variableRespuesta[indicesUnionFES],
                               family = "binomial")
  
  
  prediccionesFE <- factor(c(predict(modeloFETesteo,
                                     predictor[indicesTesteo,],
                                     type = "class")),
                           levels = c(0, 1))
  
  probFE <- c(predict(modeloFETesteo,
                      predictor[indicesTesteo, ],
                      type = "response"))
  
  prediccionesFES <- factor(c(predict(modeloFESTesteo,
                                      predictor[indicesTesteo,],
                                      type = "class")), levels = c(0, 1))
  
  probFES <- c(predict(modeloFESTesteo,
                       predictor[indicesTesteo, ],
                       type = "response"))
  
  predicciones <- data.frame(FE = prediccionesFE,
                             FES = prediccionesFES,
                             probFE = probFE,
                             probFES = probFES)
  
  ### Fase del modelo convecional
  modelo <- cv.glmnet(predictor[indicesEntrenamiento, ], 
                      variableRespuesta[indicesEntrenamiento],
                      family = "binomial")
  
  
  predicciones$NFE <- factor(c(predict(modelo,
                                       predictor[indicesTesteo,],
                                       type = "class")), levels = c(0, 1))
  
  predicciones$probNFE <- c(predict(modelo,
                                    predictor[indicesTesteo, ],
                                    type = "response"))
  
  resultadosParticion = predicciones
  resultadosParticion$Particion = i
  resultadosParticion$Posicion = indicesTesteo
  resultadosParticion$Observado = variableRespuesta[indicesTesteo]
  
  resultados <- rbind(resultados, resultadosParticion)
  
}

resultados <-
  pivot_longer(
    resultados,
    cols = c("NFE", "FE", "FES"),
    names_to = "TipoModelo",
    values_to = "Prediccion"
  )

setProb <- function(data) {
  data <- as.list(data)
  prob <- data$probNFE
  if (data$TipoModelo == "FE") {
    prob <- data$probFE  
  } else if (data$TipoModelo == "FES") {
    prob <- data$probFES
  }
  
  return(prob)
}
resultados$Prob <- as.numeric(apply(resultados, 1, setProb))
resultados <- select(resultados, -c(probFE, probFES, probNFE))

write.csv(resultados, file = "Resultados/resultadosCompletos3.csv")

resumen <- resultados %>%
  group_by(Particion, TipoModelo) %>%
  summarise(
    PCCC = PCCC(Observado, Prediccion),
    TasaError = TasaError(Observado, Prediccion),
    Sensibilidad = Sensibilidad(Observado, Prediccion),
    Especificidad = Especificidad(Observado, Prediccion),
    Matthews = mcc(Prediccion, Observado),
    Kappa = Kappa(Observado, Prediccion),
    Brier = BrierScore(Observado, Prob) 
  ) %>%
  as.data.frame()

write.csv(resumen, file = "Resultados/resumenPorParticion3.csv")


resumen <- resumen %>%
  group_by(TipoModelo) %>%
  summarise_all(~mean(.)) %>%
  as.data.frame()

resumen$Particion = NULL
write.csv(resumen, file = "Resultados/resumen3.csv")