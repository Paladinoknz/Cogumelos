import pandas as pd

# Código obtido no site onde o conjunto de dados foi disponibilizado
from ucimlrepo import fetch_ucirepo 
  
# Fetch dataset (busca o conjunto de dados)
mushroom = fetch_ucirepo(id=73) 
  
# Dados (como DataFrames do pandas)
X = mushroom.data.features 
y = mushroom.data.targets 
# Fim do código disponibilizado no site.


print("Formato da tabela X (linhas, colunas):", X.shape) # O Python sabe identificar o que é linha e coluna, então para o treinamento não precisamos definir o Y
print("Nome de todas as 22 colunas em X:\n", X.columns) # Seguindo essa mesma lógica.

# Primeiro vejo como o dataset está estruturado. Aqui podemos observar que, como os dados são categóricos, 
# precisamos usar variáveis dummy para o computador conseguir fazer a contagem e o processamento numérico.
print("\nEstruturação Padrão\n", X.head())


# Procuro por valores nulos para realizar a limpeza e o tratamento dos dados.
print("Valores nulos por coluna em X:")
print(X.isnull().sum())

# print(y.isnull().sum())


# Após a identificação da tabela que possui dados nulos, resolvi colocar como uma categoria desconhecida ('missing').
# Poderia ter usado a técnica de exclusão ou moda, mas optei por essa.
X['stalk-root'] = X['stalk-root'].fillna('missing')

# print("\n\n", X.isnull().sum())


# Após o tratamento dos valores nulos:
X_encoded = pd.get_dummies(X) # Função que organiza as variáveis dummy e faz com que o computador saiba exatamente o que cada cogumelo é através de números
X_encoded.head() # Visualização após a organização das variáveis dummy


# A partir daqui o computador consegue, matematicamente falando, ver quais cogumelos são semelhantes. 
# Escolhi a técnica de agrupamento pelo K-Means.
from sklearn.cluster import KMeans

# 1. Instanciamos o K-Means pedindo 2 grupos (k = 2)
kmeans = KMeans(n_clusters=2, random_state=42)

# 2. O algoritmo analisa os dados e descobre os centros dos 2 grupos 
kmeans.fit(X_encoded)

# 3. Guardamos o número do grupo (0 ou 1) atribuído a cada cogumelo
grupos_encontrados = kmeans.labels_

# Após isso, podemos conferir se os grupos encontrados (comestíveis e venenosos) coincidem com a variável y presente no dataset. 
# Iremos comparar e organizar com um método da biblioteca pandas.


# O squeeze foi usado porque o Python estava interpretando uma matriz de 2 dimensões e retornando um erro.
tabelaResult = pd.crosstab(y.squeeze(), grupos_encontrados, rownames=['Real (y)'], colnames=['Grupo K-Means'])
print(tabelaResult)

# O que essa tabela nos mostra: ela revela que o computador foi capaz de separar os grupamentos de maneira satisfatória, 
# visto que em momento algum ele conhecia a variável y previamente.

# Na próxima etapa agora o gabarito será entregue para que o computador aprenda supervisionado.
# Irei utilizar a técnica de árvore de decisão, que constrói uma estrutura lógica de se/senão, 
# permitindo colocar nos nós as características mais importantes, como odor, cor, etc.

# Importando bibliotecas e funções

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# 1. Separamos 70% da base para Treino e 30% para Teste.
# Pegamos o dataset e o Python já faz a seleção das partes que cada variável vai receber.
X_train, X_test, y_train, y_test = train_test_split( X_encoded, y.squeeze(), test_size=0.3, random_state=42 )



# 2. Criamos e treinamos o modelo com os dados de Treino.

# Criamos a árvore aqui, porém ainda vazia. Colocamos o 42 como semente de aleatoriedade para, quando rodar o código novamente, obter resultados consistentes.
modelo_arvore = DecisionTreeClassifier(random_state=42)

# Aqui começamos a preencher a árvore vazia, fazendo com que o modelo treine e aprenda os padrões da variável X, conferindo se está correto com a variável y.
modelo_arvore.fit(X_train, y_train)


# 3. O modelo faz as predições nos dados de TESTE (dados que ele nunca viu). 
# Como foi separado pelo Python no passo 1, o algoritmo não sabe as respostas desses dados, e aqui a gente comprova se ele aprendeu ou não.
predicoes = modelo_arvore.predict(X_test)



# 4. Medimos o percentual de acerto (Acurácia).
acuracia = accuracy_score(y_test, predicoes) 
print(f"Acurácia nos dados de teste: {acuracia * 100:.2f}%\n\n")


print(pd.crosstab(y_test, predicoes, rownames=['Real (y_test)'], colnames=['Previsto (predicoes)']))