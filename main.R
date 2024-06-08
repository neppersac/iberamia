# Carregando as bibliotecas necessárias
install.packages("caret", dependencies = TRUE)
library(caret)

install.packages("randomForest", dependencies = TRUE)
library(randomForest)

install.packages("xgboost", dependencies = TRUE)
library(xgboost)

install.packages("ggplot2", dependencies = TRUE)
library(ggplot2)

install.packages("caret", dependencies = TRUE)
library(forecast)

install.packages("moments", dependencies = TRUE)
library("moments")

library(tseries)
library(FinTS)

# 1. Definição do vetor de emissões de CO2e
# Dados de emissões de CO2e da agropecuária no Amazonas de 1990 a 2022
agropecuaria <- c(1677081, 1701660, 1674009, 1802964, 1948464, 2097143, 1864106, 1963283, 2076821, 
                  2143739, 2189929, 2255185, 2300605, 2774262, 2845493, 2937939, 3010741, 2749716, 
                  3016568, 3103937, 3141964, 3273834, 3302693, 3323927, 3196871, 2968262, 3025446, 
                  3096496, 3178738, 3348516, 3320480, 3497115, 3641670)

# 3. Criação de um vetor de anos
inicio <- 1990
final <- 2015
tamanhoTeste <- 7

# 4. Cria-se um data frame `df` com as colunas: `anos` e `agropecuaria`, 
# correspondendo aos anos e às respectivas emissões de CO2e.
df <- ts(agropecuaria, start=inicio, end = final + tamanhoTeste)

df <- data.frame(anos = as.numeric(time(df)), agropecuaria = as.numeric(df))

str(df)

ggplot(df, aes(x = anos, y = agropecuaria)) +
  geom_line(color = "black", size = 1) +   # Add the line with blue color and line width 1
  #geom_point(color = "red", size = 2) +   # Add points to the line for better visualization
  labs(x = "Year",
       y = "CO2e (t)") +    # Custom axis labels and title
  theme_minimal() 

ggplot(df, aes(x = anos, y = agropecuaria)) +
  geom_line() +
  labs(title = "Agropecuaria Emissions Over Time", x = "Year", y = "CO2e (t)") +
  theme_minimal()

# Dividindo os dados
num_train <- 26
num_test <- 7

# Cria o conjunto de treinamento e teste
train <- df[1:num_train, ]
test <- df[(num_train + 1):(num_train + num_test), ]

train
test

summary(train$agropecuaria)

# Calcular o desvio padrão
desvio_padrao <- sd(train$agropecuaria)
desvio_padrao
print(paste("Desvio padrão:", desvio_padrao))

# Calcular a curtose
curtose <- kurtosis(train$agropecuaria)
print(paste("Curtose:", curtose))

# Calcular a assimetria
assimetria <- skewness(train$agropecuaria)
print(paste("Assimetria:", assimetria))

hist(train$agropecuaria, 
     xlab = "CO2e (t)", 
     ylab = "Year", 
     main = "",
     col = "lightblue",    # Cor das barras
     border = "black")     # Cor da borda das barras

boxplot(train$agropecuaria, 
        xlab = "",         # X-Axis Label
        ylab = "CO2e (t)",   # Y-Axis Label
        main = "",
        col = "lightblue",    # Color of the box
        border = "black")     # Color of the box borders


# Modelo com tendência aditiva e erro aditivo
model_AA <- ets(train$agropecuaria, model="AAN", damped = FALSE)
# Previsões
forecast_AA <- forecast(model_AA, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_AA, test$agropecuaria)

# Modelo com tendência aditiva e erro multiplicativo
model_MA <- ets(train$agropecuaria, model="MAN", damped = FALSE)
# Previsões
forecast_MA <- forecast(model_MA, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_MA, test$agropecuaria)

# Modelo com tendência multiplicativa e erro multiplicativo
model_MM <- ets(train$agropecuaria, model="MMN", damped = FALSE)
# Previsões
forecast_MM <- forecast(model_MM, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_MM, test$agropecuaria)

# Modelo com tendência aditiva amortecida e erro aditivo
model_AAd <- ets(train$agropecuaria, model="AAN", damped = TRUE)
# Previsões
forecast_AAd <- forecast(model_AAd, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_AAd, test$agropecuaria)

# Modelo com tendência aditiva amortecida e erro multiplicativo
model_MAd <- ets(train$agropecuaria, model="MAN", damped = TRUE)
# Previsões
forecast_MAd <- forecast(model_MAd, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_MAd, test$agropecuaria)

# Modelo com tendência multiplicativa amortecida e erro multiplicativo
model_MMd <- ets(train$agropecuaria, model="MMN", damped = TRUE)
# Previsões
forecast_MMd <- forecast(model_MMd, h=length(test$agropecuaria))
# Avaliação do modelo
accuracy(forecast_MMd, test$agropecuaria)

# Dados de treino, teste e previsões
df_treino <- data.frame(ano = 1990:(1990 + length(train$agropecuaria) - 1), emissao = as.numeric(train$agropecuaria), tipo = "Treino")
df_teste <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(test$agropecuaria), tipo = "Teste")
df_prev_AA <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_AA$mean), tipo = "AA")
df_prev_MA <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_MA$mean), tipo = "MA")
df_prev_MM <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_MM$mean), tipo = "MM")
df_prev_AAd <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_AAd$mean), tipo = "AAd")
df_prev_MAd <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_MAd$mean), tipo = "MAd")
df_prev_MMd <- data.frame(ano = (1990 + length(train$agropecuaria)):(1990 + length(train$agropecuaria) + length(test$agropecuaria) - 1), emissao = as.numeric(forecast_MMd$mean), tipo = "MMd")

# Combinando todos os dados em um único data frame
df_combined <- rbind(df_treino, df_teste, df_prev_AA, df_prev_MA, df_prev_MM, df_prev_AAd, df_prev_MAd, df_prev_MMd)

# Plotando os dados combinados com legenda ajustada
ggplot(df_combined, aes(x = ano, y = emissao, color = tipo)) +
  geom_line() +
  ggtitle("Emissões de CO2e pela Agropecuária no Amazonas (1990–2022)") +
  xlab("Ano") +
  ylab("CO2e (t)") +
  labs(color = "")

# Definindo cores para as linhas
colors <- c("Treino" = "black", "Teste" = "blue", 
            "AA" = "red", "MA" = "green", 
            "MM" = "purple", "AAd" = "orange", 
            "MAd" = "brown", "MMd" = "pink")

# Plotando os dados combinados com legenda ajustada
ggplot(df_combined, aes(x = ano, y = emissao, color = tipo)) +
  geom_line() +
  ggtitle("Emissões de CO2e pela Agropecuária no Amazonas (1990–2022)") +
  xlab("Ano") +
  ylab("CO2e (t)") +
  scale_color_manual(values = colors, name = "")

# Extraindo os critérios de informação
models <- list(model_AA, model_MA, model_MM, model_AAd, model_MAd, model_MMd)
model_names <- c("AA", "MA", "MM", "AAd", "MAd", "MMd")


criteria <- data.frame(
  Model = model_names,
  AIC = sapply(models, AIC),
  AICc = sapply(models, function(m) m$aicc),
  BIC = sapply(models, BIC)
)
print(criteria)

# Função para analisar resíduos
analyze_residuals <- function(model, model_name) {
  residuals <- residuals(model)
  
  lb_test <- Box.test(residuals, lag=length(test), type="Ljung-Box")
  arch_test <- ArchTest(residuals,lags = length(test))
  jb_test <- jarque.bera.test(residuals)
  
  cat("Modelo:", model_name, "\n")
  cat("Ljung-Box test p-value:", lb_test$p.value, "\n")
  cat("ARCH test p-value:", arch_test$p.value, "\n")
  cat("Jarque-Bera test p-value:", jb_test$p.value, "\n")
  cat("\n")
}
# Analisando resíduos de cada modelo
for (i in 1:length(models)) {
  analyze_residuals(models[[i]], model_names[i])
}

# Função para calcular métricas de previsão
calculate_metrics <- function(forecast, test) {
  rmse <- sqrt(mean((forecast$mean - test)^2))
  mae <- mean(abs(forecast$mean - test))
  mape <- mean(abs((forecast$mean - test) / test)) * 100
  
  return(c(RMSE = rmse, MAE = mae, MAPE = mape))
}
# Calculando métricas para cada modelo
metrics <- data.frame(
  Model = model_names,
  do.call(rbind, lapply(1:length(models), function(i) calculate_metrics(forecast(models[[i]], h=length(test$agropecuaria)), test$agropecuaria)))
)
print(metrics)

# Supondo que 'test' seja seu conjunto de teste definido anteriormente

# Extraindo e imprimindo coeficientes de cada modelo
for (i in 1:length(models)) {
  model <- models[[i]]
  model_name <- model_names[i]
  
  # Extraindo coeficientes do modelo
  coefficients <- coef(model)
  
  cat("Modelo:", model_name, "\n")
  cat("Alpha:", coefficients["alpha"], "\n")
  if ("beta" %in% names(coefficients)) cat("Beta:", coefficients["beta"], "\n")
  if ("phi" %in% names(coefficients)) cat("Phi:", coefficients["phi"], "\n")
  if ("l" %in% names(coefficients)) cat("L:", coefficients["l"], "\n")
  if ("b" %in% names(coefficients)) cat("B:", coefficients["b"], "\n")
  cat("\n")
}

######################################################

# Treinando o modelo de Regressão Linear
modelo_lm <- lm(agropecuaria ~ anos, data = train)
modelo_lm
# Treinando o modelo de Random Forest
modelo_rf <- randomForest(agropecuaria ~ anos, data = train)
modelo_rf

str(train)
# Treinando o modelo XGBoost
modelo_xgb <- xgboost(data = as.matrix(train[, -1]), label = train$agropecuaria, nrounds = 100, objective = "reg:linear")
modelo_xgb
# Fazendo previsões com o modelo de Regressão Linear
previsoes_lm <- predict(modelo_lm, newdata = test)
previsoes_lm
# Fazendo previsões com o modelo de Random Forest
previsoes_rf <- predict(modelo_rf, newdata = test)
previsoes_rf

# Fazendo previsões com o modelo XGBoost
previsoes_xgb <- predict(modelo_xgb, as.matrix(test[, -1]))
previsoes_xgb

# Calculando as métricas de desempenho do modelo de Regressão Linear
RMSE_lm <- sqrt(mean((previsoes_lm - test$agropecuaria)^2))
MAE_lm <- mean(abs(previsoes_lm - test$agropecuaria))
MAPE_lm <- mean(abs((previsoes_lm - test$agropecuaria) / test$agropecuaria)) * 100

# Calculando as métricas de desempenho do modelo de Random Forest
RMSE_rf <- sqrt(mean((previsoes_rf - test$agropecuaria)^2))
MAE_rf <- mean(abs(previsoes_rf - test$agropecuaria))
MAPE_rf <- mean(abs((previsoes_rf - test$agropecuaria) / test$agropecuaria)) * 100

# Calculando as métricas de desempenho do modelo XGBoost
RMSE_xgb <- sqrt(mean((previsoes_xgb - test$agropecuaria)^2))
MAE_xgb <- mean(abs(previsoes_xgb - test$agropecuaria))
MAPE_xgb <- mean(abs((previsoes_xgb - test$agropecuaria) / test$agropecuaria)) * 100

# Exibindo as métricas de desempenho dos modelos
# Exibindo as métricas de desempenho dos modelos
print("Regressão Linear:")
print(paste("RMSE:", RMSE_lm))
print(paste("MAE:", MAE_lm))
print(paste("MAPE:", MAPE_lm))

print("Random Forest:")
print(paste("RMSE:", RMSE_rf))
print(paste("MAE:", MAE_rf))
print(paste("MAPE:", MAPE_rf))

print("XGBoost:")
print(paste("RMSE:", RMSE_xgb))
print(paste("MAE:", MAE_xgb))
print(paste("MAPE:", MAPE_xgb))

# Criando um dataframe com as previsões de todos os modelos
df_previsoes <- data.frame(
  ano = c(test$anos, test$anos, test$anos),
  energia = c(previsoes_lm, previsoes_rf, previsoes_xgb),
  modelo = rep(c("Regressão Linear", "Random Forest", "XGBoost"), each = length(test$ano))
)

# Plotando o gráfico
ggplot() +
  geom_line(data = df, aes(x = anos, y = agropecuaria, color = "Original"), size = 1) +
  geom_line(data = df_previsoes, aes(x = ano, y = energia, color = modelo), size = 1, linetype = "dashed") +
  labs(title = "Previsões de Emissões de CO2e da Produção de Energia",
       x = "Ano",
       y = "Emissões de CO2e",
       color = "Modelo") +
  theme_minimal()


# Ajustando o dataframe para plotagem
df_plot <- rbind(
  data.frame(ano = train$anos, agropecuaria = train$agropecuaria, tipo = "Treino"),
  data.frame(ano = test$anos, agropecuaria = test$agropecuaria, tipo = "Teste")
)

ggplot() +
  geom_line(data = df_plot, aes(x = ano, y = agropecuaria, color = tipo), size = 1) +
  geom_line(data = df_previsoes, aes(x = ano, y = energia, color = modelo), size = 1, linetype = "dashed") +
  labs(title = "Previsões de Emissões de CO2e da Produção de Energia",
       x = "Ano",
       y = "Emissões de CO2e",
       color = "Tipo/Modelo") +
  theme_minimal() +
  scale_color_manual(values = c("Treino" = "black", "Teste" = "blue", "Regressão Linear" = "red", "Random Forest" = "purple", "XGBoost" = "green"))


################################################

melhor_arima <- function(serie, treino, teste, titulo, exibir, diferenca = 1){
  treino <- treino
  teste <- teste
  n <- length(treino)
  
  # Inicializar valores mínimos de AIC e BIC como infinito
  min_aic <- Inf
  menorRMSE <- Inf
  melhor_modelo <- NULL
  
  confianca <- 0.95
  
  DRIFT <- c(FALSE, TRUE)
  CONST <- c(FALSE, TRUE)
  AR <- 2
  I <- diferenca
  MA <- 2
  if (titulo == "MUTF" || titulo == "Total"){
    AR <- 3
    I <- 2
    MA <- 3
  }
  
  d <- I
  print(titulo)
  print("Valor de d: ")
  print(d)
  # Loop para testar todas as combinações
  for (t in DRIFT) {
    for (c in CONST) {
      for (p in 0:AR) {
        for (q in 0:MA) {
          modelo <- Arima(treino, order = c(p, d, q), include.drift = t, include.constant = c, method = "ML", include.mean = TRUE)
          
          aic <- AIC(modelo)
          
          k <- length(coef(modelo))
          aicc <- aic + (2 * k * (k + 1)) / (n - k - 1)
          bic <- BIC(modelo)
          residuos <- residuals(modelo)
          
          ljungbox <- Box.test(residuos,lag=length(teste),type="Ljung-Box")
          jarquebera <- jarque.bera.test(residuos)
          arch <- ArchTest(residuos,lags = length(teste))
          
          # Previsões
          previsoes <- forecast(modelo, length(teste))
          acuracia <- accuracy(previsoes, teste)
          
          #if (aic < min_aic && jarque.bera.test(residuals(modelo_arima))$p.value > 0) {
          if (aic < min_aic) {
            min_aic <- aic
            melhor_modelo_AIC <- modelo
          }
          if (acuracia[2, 'RMSE'] < menorRMSE){
            menorRMSE <- acuracia[2, 'RMSE']
            melhor_modelo_RMSE <- modelo
          }
          if (exibir){
            print(glue::glue("ARIMA({p},{d},{q}), DRIFT: {t}, CONST: {c}   AIC: {aic}, AICc: {aicc}, bic: {bic}, ljungbox: {ljungbox$p.value}, jarquebera: {jarquebera$p.value}, arch: {arch$p.value}, rsme: {acuracia[2, 'RMSE']}, mae: {acuracia[2, 'MAE']}, mape: {acuracia[2, 'MAPE']}"))
          }
        } # fim do for do q
        
      } # fim do for do p
    } # fim do for do c
  } # fim do for do d
  
  # Retornar o melhor modelo
  resultado <- list(melhor_modelo_AIC = melhor_modelo_AIC, melhor_modelo_RMSE = melhor_modelo_RMSE)
  return(resultado)
}

serie <- df$agropecuaria
treino <- train$agropecuaria
teste <- test$agropecuaria

resultado <- melhor_arima(serie, treino, teste, "Agropecuaria", TRUE)
resultado

###########################
# Aplicando a diferença de primeira ordem na série
serie_diferenciada <- diff(train$agropecuaria, differences = 1)

# Plotando o gráfico de ACF
acf(serie_diferenciada, main = "Função de Autocorrelação (ACF) das Emissões de CO2e da Agropecuária")

# Plotando o gráfico de PACF
pacf(serie_diferenciada, main = "Função de Autocorrelação Parcial (PACF) das Emissões de CO2e da Agropecuária")


# Extraindo o melhor modelo ARIMA com base no critério RMSE
melhor_modelo_arima <- resultado$melhor_modelo_RMSE

# Fazendo previsões com o melhor modelo ARIMA
previsoes_arima <- forecast(melhor_modelo_arima, h=length(teste))

# Criando um dataframe com as previsões do modelo ARIMA
df_prev_arima <- data.frame(
  ano = (max(train$anos) + 1):(max(train$anos) + length(teste)),
  agropecuaria = as.numeric(previsoes_arima$mean),
  tipo = "ARIMA"
)

# Combinando todos os dados (treinamento, teste e previsões)
df_combined_arima <- rbind(
  data.frame(ano = train$anos, agropecuaria = train$agropecuaria, tipo = "Treino"),
  data.frame(ano = test$anos, agropecuaria = test$agropecuaria, tipo = "Teste"),
  df_prev_arima
)

# Definindo cores para as linhas
colors_arima <- c("Treino" = "black", "Teste" = "blue", "ARIMA" = "red")

# Plotando os dados combinados com as previsões do modelo ARIMA
ggplot(df_combined_arima, aes(x = ano, y = agropecuaria, color = tipo)) +
  geom_line() +
  ggtitle("Previsões de Emissões de CO2e da Agropecuária no Amazonas (1990–2022)") +
  xlab("Ano") +
  ylab("Emissões de CO2e") +
  scale_color_manual(values = colors_arima, name = "Tipo")

##########################################

# Extraindo o melhor modelo de suavização exponencial com base no menor RMSE
melhor_modelo_suavizacao <- model_MM  # Supondo que model_AA seja o melhor modelo com base nas métricas calculadas anteriormente

# Fazendo previsões com o melhor modelo de suavização exponencial
previsoes_suavizacao <- forecast(melhor_modelo_suavizacao, h=length(teste))

# Criando um dataframe com as previsões do modelo de suavização exponencial
df_prev_suavizacao <- data.frame(
  ano = (max(train$anos) + 1):(max(train$anos) + length(teste)),
  agropecuaria = as.numeric(previsoes_suavizacao$mean),
  tipo = "ETS"
)

# Criando um dataframe com as previsões do modelo ARIMA
df_prev_arima <- data.frame(
  ano = (max(train$anos) + 1):(max(train$anos) + length(teste)),
  agropecuaria = as.numeric(previsoes_arima$mean),
  tipo = "ARIMA"
)

# Combinando todos os dados (treinamento, teste e previsões)
df_combined_final <- rbind(
  data.frame(ano = train$anos, agropecuaria = train$agropecuaria, tipo = "Train"),
  data.frame(ano = test$anos, agropecuaria = test$agropecuaria, tipo = "Test"),
  df_prev_suavizacao,
  df_prev_arima
)

# Definindo cores para as linhas
colors_final <- c("Train" = "black", "Test" = "blue", "ETS" = "green", "ARIMA" = "red")

# Plotando os dados combinados com as previsões dos melhores modelos
ggplot(df_combined_final, aes(x = ano, y = agropecuaria, color = tipo)) +
  geom_line() +
  ggtitle("Previsões de Emissões de CO2e da Agropecuária no Amazonas (1990–2022)") +
  xlab("Year") +
  ylab("CO2e (t)") +
  scale_color_manual(values = colors_final, name = "")

# Aplicando a diferença de primeira ordem na série
serie_diferenciada <- diff(train$agropecuaria, differences = 1)

# Calculando e plotando o gráfico de ACF sem o tempo zero
acf_values <- acf(serie_diferenciada, plot = FALSE)
acf_values$lag <- acf_values$lag[-1]  # Removendo o lag zero
acf_values$acf <- acf_values$acf[-1]  # Removendo o valor de ACF correspondente ao lag zero

# Plotando o gráfico de ACF atualizado
plot(acf_values, main = "Função de Autocorrelação (ACF) das Emissões de CO2e da Agropecuária")
