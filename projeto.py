# --- IMPORTAÇÕES PARA ANÁLISE DE SENTIMENTO ---
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Garantir recursos NLTK
for recurso in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{recurso}' if recurso=='punkt' else f'corpora/{recurso}')
    except LookupError:
        nltk.download(recurso)

# Stopwords portuguesas + limpeza de acentos
stop_words = set(stopwords.words('portuguese'))

def limpar_texto(texto):
    """Remove acentuação e caracteres especiais"""
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç]+', ' ', texto)
    return texto

def extrair_palavras_chave(texto):
    """Tokeniza, normaliza e remove stopwords"""
    texto_limpo = limpar_texto(texto)
    palavras = word_tokenize(texto_limpo)
    palavras_chave = [p for p in palavras if p.isalpha() and p not in stop_words]
    return palavras_chave

# Configurar pipeline de sentimento (modelo português confiável)
try:
    analisador_sentimento_pt = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )
except Exception as e:
    print(f"AVISO: Falha ao carregar modelo: {e}")
    analisador_sentimento_pt = None

def analisar_sentimento_transformers(texto):
    """Analisa sentimento e retorna polaridade (-1 a 1)"""
    if analisador_sentimento_pt is None:
        return "MODELO INDISPONÍVEL"
    
    resultado = analisador_sentimento_pt(texto)[0]
    label = resultado['label']
    score = resultado['score']
    
    if label == 'POS':
        return score
    elif label == 'NEG':
        return -score
    else:  # NEU
        return 0.0

# Exemplos
textos = [
    "Eu adorei o atendimento do restaurante!",
    "O pedido demorou muito para chegar.",
    "A comida estava deliciosa e bem apresentada."
]

for t in textos:
    palavras = extrair_palavras_chave(t)
    sentimento = analisar_sentimento_transformers(t)
    
    print(f"Texto: {t}")
    print(f"Palavras-chave: {palavras}")
    if isinstance(sentimento, str):
        print(f"Sentimento: {sentimento}\n")
    else:
        print(f"Sentimento: {sentimento:.4f}\n")
