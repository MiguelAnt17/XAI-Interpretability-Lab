# XAI - LIME: Detecção de Fake News em Mensagens de WhatsApp 

Este projeto aplica técnicas de **Inteligência Artificial Explicável (XAI)**, especificamente **LIME (Local Interpretable Model-agnostic Explanations)**, para analisar e classificar mensagens de Fake News do WhatsApp em Português.

O objetivo principal não é apenas classificar as mensagens com precisão, mas também **entender e explicar** por que os modelos tomam determinadas decisões, identificando palavras-chave e padrões linguísticos que influenciam na detecção de desinformação.

## Funcionalidades Principais

*   **Pré-processamento Avançado de Texto**: Pipeline robusto que trata emojis, gírias de internet (ex: "kkk"), pontuação, stopwords e realiza lematização.
*   **Treinamento Multi-modelo**: Comparação de desempenho de diversos algoritmos de classificação:
    *   Logistic Regression
    *   Random Forest
    *   Support Vector Machine (SVM)
    *   Multi-Layer Perceptron (MLP/Redes Neurais)
    *   Naive Bayes (Bernoulli e Complement)
*   **Explicabilidade com LIME**: Geração de explicações locais para entender quais palavras contribuíram para a classificação de uma mensagem como "Fake" ou "Verdadeira".
*   **Análise de Viés (Bias Analysis)**: Visualização gráfica da discrepância de desempenho (F1-score) entre classes para identificar vieses nos modelos.
*   **Métricas de Confiança**: Avaliação da estabilidade e fidelidade das explicações geradas pelo LIME.

## Estrutura do Repositório

*   `experiments_final.ipynb`: Notebook Jupyter principal contendo todo o fluxo de execução, desde o carregamento dos dados até a geração de explicações e gráficos.
*   `utils.py`: Módulo Python com funções auxiliares encapsuladas para limpeza de texto, vetorização, avaliação de modelos e visualização.
*   `fakeWhatsApp.BR_2018.csv`: Dataset utilizado (mensagens de WhatsApp coletadas no Brasil em 2018).
*   `requirements.txt`: Lista de dependências do projeto.
*   `outputs/`: (Gerado na execução) Contém relatórios de classificação, gráficos e logs.

## Instalação e Uso

### Pré-requisitos

Certifique-se de ter o Python 3.8+ instalado. Recomenda-se o uso de um ambiente virtual.

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

# Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### Executando o Projeto

1.  Certifique-se de que o arquivo de dados `fakeWhatsApp.BR_2018.csv` está no diretório raiz.
2.  Abra o notebook principal:
    ```bash
    jupyter notebook experiments_final.ipynb
    ```
3.  Execute as células sequencialmente para reproduzir os experimentos.

## Model Card - Counterfactual Explanations
---
language:
- pt
license: mit
tags:
- ptt5
- xai
- counterfactuals
- polyjuice
- synthetic-data
datasets:
- assin2
base_model: unicamp-dl/ptt5-base-portuguese-vocab
---

### PTT5-Student for Counterfactual Generation (Portuguese)

Este modelo é um **gerador de explicações contrafactuais** para a língua portuguesa. É uma versão *fine-tuned* do modelo [PTT5-Base](https://huggingface.co/unicamp-dl/ptt5-base-portuguese-vocab), treinado utilizando a técnica de *Knowledge Distillation*.

O modelo atua como um "Student", tendo aprendido a gerar perturbações textuais a partir de um dataset sintético criado pelo **Gemini 1.5 Flash** (Teacher), seguindo a taxonomia de controlo do método *Polyjuice*.

## 🎯 Intended Use
Este modelo foi desenvolvido no âmbito de uma Tese de Mestrado sobre **Causalidade e XAI (Explainable AI)**. O seu objetivo é servir como mecanismo de perturbação para métodos de explicabilidade, gerando variações de frases baseadas em códigos de controlo.

**Códigos Suportados:**
* `[negation]`: Adiciona/remove negação.
* `[quantifier]`: Altera quantidades/números.
* `[lexical]`: Substitui palavras por sinónimos/antónimos.
* `[insert]`: Adiciona informação/adjetivos.
* `[delete]`: Remove informação não essencial.
* `[restructure]`: Altera a voz ou estrutura sintática.

## How to Use

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "oteuuser/ptt5-polyjuice-student" # Substituir pelo teu path
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input format: "gerar contrafactual [codigo]: frase"
input_text = "gerar contrafactual [negation]: O ministro assinou o decreto ontem."

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids, 
    max_length=128, 
    num_beams=5, 
    early_stopping=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output esperado: "O ministro não assinou o decreto ontem."
```

## Resultados e Visualizações

O projeto gera diversas visualizações para auxiliar na interpretação:

*   **Matrizes de Confusão**: Para avaliar erros e acertos por classe.
*   **Gráficos de Explicação LIME**: Destacam palavras positivas/negativas para a predição.
*   **Gráfico de Viés**: Compara o F1-Score entre classes reais e fake para todos os modelos.
*   **Ranking de Palavras**: Identifica os termos mais frequentes e influentes em cada classe.

## Autor

**Miguel Maurício António**
*   Projeto desenvolvido no contexto de pesquisa em NLP e XAI.
