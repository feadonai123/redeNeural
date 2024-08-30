import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.datasets as dataset
import threading

np.random.seed(8)

semaphore = threading.Semaphore()

print("\n\nINICIO\n\n")

def getCSVDataset(filename):
  df = pd.read_csv(filename)
  x = df.iloc[:, :-1].values
  y = df.iloc[:, -1].values
  return normalize(x, y)

def partitionDataset(x, y, train_size=0.8):
  indices = np.random.permutation(x.shape[0])
  train_len = int(x.shape[0] * train_size)
  
  x_train = x[indices[:train_len]]
  y_train = y[indices[:train_len]]
  
  x_test = x[indices[train_len:]]
  y_test = y[indices[train_len:]]
  
  return x_train, y_train, x_test, y_test
  
def normalize(x, y):
  y = y - 1
  x = (x - x.mean()) / x.std()
  return x, y

def denormalize(x, y):
  y = y + 1
  x = (x * x.std()) + x.mean()
  return x, y
  
def get2DDataset():
  x, y = dataset.make_moons(n_samples=500, noise=0.1)
  return (x, y)

def get3DDataset():
  x, y = dataset.make_blobs(n_samples=500, n_features=3, centers=4, random_state=70, cluster_std=1.5, shuffle=True)
  return (x, y)

def showGraph(x, y, title="Graph", onlyDownload=False, label=None):
  if x.shape[1] == 2:
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.5)
    plt.title(title)
    
    if onlyDownload:
      plt.savefig(f"./results/{label}.png")
      plt.close()
    else:
      plt.show()
  elif x.shape[1] == 3:
    fig = plt.figure()
    
    ax1 = fig.add_subplot(projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, alpha=0.5)
    ax1.set_title(title)
    
    if onlyDownload:
      plt.savefig(f"./results/{label}.png")
      plt.close()
    else:
      plt.show()
  
      
  
def showAcuracyGraph(accuracies, title="Accuracy", onlyDownload=False, label=None):
  plt.plot(accuracies)
  plt.title(title)
  print("chegou aqui")
  
  if onlyDownload:
    print('vai salvar')
    plt.savefig(f"./results/{label}.png")
    plt.close()
  else:
    print('vai mostrar')
    plt.show()
    
  print("saiu")
  
def showComparisonGraph(x, y, predictions, subtitle=None, onlyDownload=False, label=None):
  if x.shape[1] == 2:
    fig, axs = plt.subplots(1, 2, figsize=(16, 12))
    axs[0].scatter(x[:, 0], x[:, 1], c=y, alpha=0.5, cmap='cool')
    axs[0].set_title("Real")
    
    axs[1].scatter(x[:, 0], x[:, 1], c=predictions, alpha=0.5, cmap='cool')
    axs[1].set_title("Predictions")
    
    fig.suptitle(subtitle)
    if onlyDownload:
      plt.savefig(f"./results/{label}.png")
      plt.close()
    else:
      plt.show()
  elif x.shape[1] == 3:
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, alpha=0.5, cmap='cool')
    ax1.set_title("Real")
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x[:, 0], x[:, 1], x[:, 2], c=predictions, alpha=0.5, cmap='cool')
    ax2.set_title("Predictions")
    
    fig.suptitle(subtitle)
    if onlyDownload:
      plt.savefig(f"./results/{label}.png")
      plt.close()
    else:
      plt.show()

def printDataset(x, y, show_graph=False):
  df = pd.DataFrame(np.concatenate((x, y.reshape(-1, 1)), axis=1))

  unique = np.unique(y, return_counts=True)

  print("Dataset")
  print(df)

  print("\n")
  print("Informações do Dataset")
  print(f"{x.shape =}")
  print(f"{y.shape =}")
  
  print("\n")
  print("Classes disponíveis")
  for label, count in zip(unique[0], unique[1]):
      print(f"Classe {label}\tAmostra: {count}")
      
  if show_graph:
    showGraph(x, y)
      
class RedeNeural:
  def __init__(self, x: np.array, y: np.array, hidden_neurons: int = 10):
    self.x = x
    self.y = y
    self.hidden_neurons = hidden_neurons
    self.output_neurons = len(np.unique(y))
    self.input_neurons = self.x.shape[1]
    
    # Pesos ligando inputs com a camada oculta
    self.W1 = np.random.randn(self.input_neurons, self.hidden_neurons)
    self.B1 = np.zeros((1, self.hidden_neurons))
    
    self.W2 = np.random.randn(self.hidden_neurons, self.output_neurons)
    self.B2 = np.zeros((1, self.output_neurons))
    
    self.model_dict = {
      'W1': self.W1,
      'B1': self.B1,
      'W2': self.W2,
      'B2': self.B2
    }
  
  def forward(self, x: np.array) -> np.array:
    # calculo da camada entre input e hidden
    self.z1 = np.dot(x, self.W1) + self.B1
    self.f1 = self.tanh(self.z1)
    
    # calculo da camada entre hidden e output
    z2 = np.dot(self.f1, self.W2) + self.B2
    
    # softmax: A soma das saídas de cada neurônio é 1
    exp_values = np.exp(z2)
    softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return softmax
  
  def sigmoid(self, x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))
  
  def tanh(self, x: np.array) -> np.array:
    return np.tanh(x)
  
  def loss(self, softmax: np.array):
    previsoes = np.zeros(self.y.shape)
    
    for i, classe in enumerate(self.y):
      previsto = softmax[i][classe]
      previsoes[i] = previsto
      
    # -Y' * log(Y)
    # Y' é a saida real
    # Y é a saida prevista
    # Y' SEMPRE será 1, pois é a classe real
    log_prob = (- 1 * np.log(previsoes)).sum()
    return log_prob / self.y.shape[0]
  
  def backpropagation(self, softmax: np.ndarray, learning_rate: float = 0.01) -> None:
    delta2 = np.copy(softmax)
    # Diminui 1 da probabilidade prevista da classe real
    delta2[range(self.y.shape[0]), self.y] -= 1
    
    # Calcula derivadas para camada 2
    dW2 = np.dot(self.f1.T, delta2)
    dB2 = np.sum(delta2, axis=0, keepdims=True)
    
    delta1 = np.dot(delta2, self.W2.T) * (1 - np.power(np.tanh(self.z1), 2))
    dW1 = np.dot(self.x.T, delta1)
    dB1 = np.sum(delta1, axis=0, keepdims=True)
    
    # Atualiza pesos e bias
    self.W1 += -learning_rate * dW1
    self.B1 += -learning_rate * dB1
    self.W2 += -learning_rate * dW2
    self.B2 += -learning_rate * dB2
    
  def plotGraph(self, predictions, epoch):
    
    print(f"{self.x.shape =}")
    if self.x.shape[1] == 2:
      fig, axs = plt.subplots(1, 2, figsize=(16, 12))
      axs[0].scatter(self.x[:, 0], self.x[:, 1], c=self.y, alpha=0.5, cmap='cool')
      axs[0].set_title("Real")

      axs[1].scatter(self.x[:, 0], self.x[:, 1], c=predictions, alpha=0.5, cmap='cool')
      axs[1].set_title("Predictions")
      
      # correctPredictions = (predictions == y)
      # wrongPredictions = (predictions != y)
      
      # axs[1].scatter(x[correctPredictions][:, 0], x[correctPredictions][:, 1], c=predictions[correctPredictions], alpha=0.5)
      # axs[1].scatter(x[wrongPredictions][:, 0], x[wrongPredictions][:, 1], c='red', alpha=0.5)

      fig.suptitle(f"Accuracy: {self.accuracy:.3f} - Loss: {self.error:.3f} - Hidden Neurons: {self.hidden_neurons} - Output Neurons: {self.output_neurons} - Epoch: {epoch}")
      plt.show()
    elif self.x.shape[1] == 3:
      fig = plt.figure()
      
      ax1 = fig.add_subplot(projection='3d')
      ax1.scatter(self.x[:, 0], self.x[:, 1], self.x[:, 2], c=self.y, alpha=0.5, cmap='cool')
      ax1.set_title("Real")

      # Adicionar o segundo subplot em 3D
      ax2 = fig.add_subplot(projection='3d')
      ax2.scatter(self.x[:, 0], self.x[:, 1], self.x[:, 2], c=predictions, alpha=0.5, cmap='cool')
      ax2.set_title("Predictions")
      
      fig.suptitle(f"3D Accuracy: {self.accuracy:.3f} - Loss: {self.error:.3f} - Hidden Neurons: {self.hidden_neurons} - Output Neurons: {self.output_neurons} - Epoch: {epoch}")
      
      plt.show()
  
  def fit(self, epochs: int = 1000, learning_rate: float = 0.01, show_graph: bool = False) -> np.array:
    print("\n\nINICIO DO TREINAMENTO\n\n")
    self.accuracy_history = []
    for epoch in range(epochs):
      outputs = self.forward(self.x)
      self.error = self.loss(outputs)
      self.backpropagation(outputs, learning_rate)
      
      
      # Acuracia
      predictions = np.argmax(outputs, axis=1)
      
      correct_predictions = (predictions == self.y).sum()
      self.accuracy = correct_predictions / self.y.shape[0]
      self.accuracy_history.append(self.accuracy)
      
      if int((epoch) % (epochs / 20)) == 0 or epoch == epochs - 1:
        if show_graph:
          self.plotGraph(predictions, epoch)
        print(f"Epoch [{epoch} / {epochs}]\t Loss: {self.error.item():.4f}\t Accuracy: {self.accuracy:.3f}")
        
        if self.accuracy == 1:
          break
        
      
    print("\n\nFIM DO TREINAMENTO\n\n")
    
    return predictions

hidden_neurons = 18
learning_rate = 0.0008
epochs = 70000

x_all, y_all = getCSVDataset("dataset3.csv")
# x_all, y_all = get3DDataset()
printDataset(x_all, y_all, show_graph=True)

x_train, y_train, x_test, y_test = partitionDataset(x_all, y_all, train_size=0.8)


modelo = RedeNeural(x_train, y_train, hidden_neurons=hidden_neurons)
print("\n============INICIANDO TREINAMENTO======================")
result = modelo.fit(epochs=epochs, learning_rate=learning_rate, show_graph=False)
print("============FIM TREINAMENTO======================")



print("\n============RESULTADOS TREINAMENTO======================")
print(f"{x_train.shape =}")
print(f"{modelo.accuracy =}")
print(f"{modelo.error =}")
print(f"{hidden_neurons =}")
print(f"{learning_rate =}")
print(f"{epochs =}")

subtitle = f"[TREINO {x_train.shape[0]} amostras]\n\nAcurácia: {modelo.accuracy:.3f} - Loss: {modelo.error:.3f}\n\nNeurônios na camada oculta: {hidden_neurons} - Épocas: {epochs} - lr: {learning_rate}"
showComparisonGraph(x_train, y_train, result, subtitle=subtitle)
showAcuracyGraph(modelo.accuracy_history, title=subtitle)
print("==================================")


# Validação do modelo
softmax = modelo.forward(x_test)
result = np.argmax(softmax, axis=1)

correct_predictions = (result == y_test).sum()
acuracy_test = correct_predictions / y_test.shape[0]
print("\n============RESULTADOS TESTE======================")
print(f"{x_test.shape =}")
print(f"{acuracy_test =}")
print(f"{hidden_neurons =}")

subtitle = f"[TESTE {x_test.shape[0]} amostras]\n\nAcurácia: {acuracy_test:.3f}"
showComparisonGraph(x_test, y_test, result, subtitle=subtitle)
print("==================================")
print("\n\nFIM\n\n")
