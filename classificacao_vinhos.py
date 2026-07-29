# 1. Importando as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# 2. Carregando os dados
caminho = r"C:\Users\Administrador\Documents\Python 2026\Pandas\wine_dataset.csv"
arquivo = pd.read_csv(caminho)

print("--- Primeiras 5 linhas do dataset ---")
print(arquivo.head())

# 3. Mapeando a coluna 'style' (red -> 0, white -> 1)
arquivo["style"] = arquivo["style"].map({"red": 0, "white": 1})

# 4. Separando as variáveis x (recursos) e y (resultado que queremos prever)
y = arquivo["style"]
x = arquivo.drop("style", axis=1)

# 5. Criando os conjuntos de treino e teste (30% para teste)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

print(f"\nQuantidade de linhas para teste: {x_teste.shape[0]}")

# 6. Criando e treinando o modelo
modelo = ExtraTreesClassifier()
modelo.fit(x_treino, y_treino)

# 7. Testando a precisão do modelo
resultado = modelo.score(x_teste, y_teste)
print(f"\nAcurácia do modelo: {resultado * 100:.2f}%")

# 8. Testando previsões em um pequeno pedaço dos dados (amostra de 5 vinhos)
amostra_x = x_teste[600:605]
amostra_y_real = y_teste[600:605]

previsao = modelo.predict(amostra_x)

print("\n--- Comparando o resultado Real vs Previsão ---")
print("Valores reais:   ", amostra_y_real.values)
print("Previsão do modelo:", previsao)