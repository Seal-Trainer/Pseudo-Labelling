BrierScore <- function(observado, probPredicha)
{
  if(is.factor(observado))
  {
    observado <- as.numeric(observado) - 1
  }
  
  mean((probPredicha-observado)^2)
}

Kappa <- function(observado, predicho)
{
  confusionMatrix(table(observado, predicho))$overall['Kappa']
}

Sensibilidad <- function(observado, predicho)
{
  confusionMatrix(table(observado, predicho))$byClass['Sensitivity']
}

Especificidad <- function(observado, predicho)
{
  confusionMatrix(table(observado, predicho))$byClass['Specificity']
}

PCCC <- function(observado, predicho)
{
  mean(observado == predicho)
}

TasaError <- function(observado, predicho)
{
  1 - PCCC(observado, predicho)
}

