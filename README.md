# Análise de Sentimento em Português 🇧🇷

Este projeto realiza **extração de palavras-chave** e **análise de sentimento** de textos em português usando Python, NLTK e Transformers.

---

## Funcionalidades

- Limpeza de texto (remoção de acentuação e caracteres especiais)
- Tokenização e remoção de stopwords
- Extração de palavras-chave
- Análise de sentimento com modelo `pysentimiento/robertuito-sentiment-analysis`
- Retorno de polaridade entre -1 (negativo) e 1 (positivo)

---

## Pré-requisitos

- Python 3.8 ou superior
- Bibliotecas:

```bash
pip install nltk transformers torch
