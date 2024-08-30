import numpy as np

class RedeNeural:
    def __init__(self, num_entradas, num_ocultas, num_saidas):
        # Inicializar os pesos e bias de forma aleatória
        self.pesos_entrada_oculta = np.random.randn(num_entradas, num_ocultas)
        self.bias_oculta = np.zeros((1, num_ocultas))
        self.pesos_oculta_saida = np.random.randn(num_ocultas, num_saidas)
        self.bias_saida = np.zeros((1, num_saidas))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return x * (1 - x)

    def forward(self, entrada):
        # Camada Oculta
        self.z_oculta = np.dot(entrada, self.pesos_entrada_oculta) + self.bias_oculta
        self.a_oculta = self.sigmoid(self.z_oculta)

        # Camada de Saída
        self.z_saida = np.dot(self.a_oculta, self.pesos_oculta_saida) + self.bias_saida
        self.a_saida = self.sigmoid(self.z_saida)

        return self.a_saida

    def backward(self, saida_calculada, saida_esperada, taxa_aprendizado):
        # Calcular o erro da camada de saída
        erro_saida = saida_calculada - saida_esperada
        derivada_erro_saida = erro_saida * self.derivada_sigmoid(saida_calculada)

        # Calcular o erro da camada oculta
        derivada_erro_oculta = np.dot(derivada_erro_saida, self.pesos_oculta_saida.T) * self.derivada_sigmoid(self.a_oculta)

        # Ajustar os pesos e bias
        self.pesos_oculta_saida -= taxa_aprendizado * np.dot(self.a_oculta.T, derivada_erro_saida)
        self.bias_saida -= taxa_aprendizado * np.sum(derivada_erro_saida, axis=0, keepdims=True)
        self.pesos_entrada_oculta -= taxa_aprendizado * np.dot(entrada.T, derivada_erro_oculta)
        self.bias_oculta -= taxa_aprendizado * np.sum(derivada_erro_oculta, axis=0, keepdims=True)

    def treinar(self, entrada, saida_esperada, epochs, taxa_aprendizado):
        for epoch in range(epochs):
            saida_calculada = self.forward(entrada)
            self.backward(saida_calculada, saida_esperada, taxa_aprendizado)

            # Imprimir o erro a cada epoch (opcional)
            erro_total = np.sum((saida_esperada - saida_calculada) ** 2)
            print(f"Epoch {epoch+1}: Erro = {erro_total}")

# Criar a rede neural
rede = RedeNeural(2, 3, 1)  # 2 entradas, 3 neurônios na camada oculta, 1 saída

# Dados de Treinamento
entrada = np.array([[0.1, 0.2]])
saida_esperada = np.array([[0.9]])

# Treinar a rede
rede.treinar(entrada, saida_esperada, epochs=1000, taxa_aprendizado=0.1)

# Predizer a saída para novos dados (opcional)
nova_entrada = np.array([[0.3, 0.4]])
saida_predita = rede.forward(nova_entrada)
print(f"Saída Predita: {saida_predita}") 
