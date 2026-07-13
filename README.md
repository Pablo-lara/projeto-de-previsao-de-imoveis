# Sistema de Previsão de Preços de Imóveis

## Sobre o projeto

Este projeto utiliza técnicas de Machine Learning para prever o preço de imóveis com base em características como localização, renda média da região, quantidade de cômodos, população e idade das construções.

O modelo foi desenvolvido utilizando Python e Scikit-Learn, seguindo as principais etapas de um fluxo de trabalho de Ciência de Dados.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

## Funcionalidades

- Carregamento e análise dos dados
- Limpeza e preparação do dataset
- Separação dos dados em treino e teste
- Treinamento de um modelo de Regressão Linear
- Avaliação utilizando MAE, RMSE e R²
- Salvamento do modelo treinado
- Previsão do preço de novos imóveis

## Estrutura do projeto

```
projeto-previsao-imoveis/
│
├── data/
│   └── housing.csv
│
├── app.py
├── modelo_previsao_imoveis.pkl
├── requirements.txt
```

## Como executar

1. Clone este repositório.

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute o projeto:

```bash
python app.py
```

## Dataset

California Housing Dataset

## Autor

Pablo V. Lara
