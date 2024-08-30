import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def derivada_sigmoid(x):
  return x * (1 - x)

def perceptron(inputs, weights, bias):
  soma = 0
  for i in range(len(inputs)):
    soma += inputs[i] * weights[i]
  return sigmoid(soma + bias)

def erro(saida_calculada, saida_esperada):
  soma = 0
  for i in range(len(saida_calculada)):
    soma += (saida_calculada[i] - saida_esperada[i]) ** 2
  return soma

def gradiente_local(saida_calculada, saida_esperada):
  return 2 * (saida_calculada - saida_esperada) * derivada_sigmoid(saida_calculada)

# primeira posicao do array de pesos é o bias
pesos = [
  [
    [0, 0.2, 0.3],
    [0, 0.4, 0.5]
  ],
  [
    [0, 0.6, 0.7],
  ]
]
entrada = [0.1, 0.2]
saida_esperada = [0.9]
taxa_aprendizado = 0.1


for i in range(len(pesos)):
  if i == 0:
    inputs = entrada
  else:
    inputs = saida_calculada

  saida_calculada = []
  for j in range(len(pesos[i])):
    saida_calculada.append(perceptron(inputs, pesos[i][j][1:], pesos[i][j][0]))
    

erro_total = erro(saida_calculada, saida_esperada)

for i in range(len(pesos) - 1, -1, -1):

  for j in range(len(pesos[i])):
    
    gradiente = gradiente_local(saida_calculada[j], saida_esperada[j])
    
    for k in range(len(pesos[i][j])):
      if k == 0:
        pesos[i][j][k] -= taxa_aprendizado * gradiente[j]
      else:
        pesos[i][j][k] -= taxa_aprendizado * gradiente[j] * inputs[k - 1]
    
