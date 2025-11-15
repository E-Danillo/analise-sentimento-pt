# --- Importações ---
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Garantir recursos NLTK ( Processamento de linguagem natural )
for recurso in ['punkt', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{recurso}' if recurso=='punkt' else f'corpora/{recurso}')
    except LookupError:
        nltk.download(recurso)

# Palavras que vamos ignorar na análise ( Stopwords portuguesas )
stop_words = set(stopwords.words('portuguese'))

# Função de remover pontuação, caracteres especiais e tudo minúsculo
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúç]+', ' ', texto)
    return texto

# Função de extrair palavras-chave do texto
def extrair_palavras_chave(texto):
    texto_limpo = limpar_texto(texto)
    palavras = word_tokenize(texto_limpo)
    palavras_chave = [p for p in palavras if p.isalpha() and p not in stop_words]
    return palavras_chave

# Tenta criar um modelo pronto pra analisar sentimento de textos em português. Se falhar (internet ruim ou PC fraco), mostra aviso.
try:
    analisador_sentimento_pt = pipeline(
        "sentiment-analysis",
        model="pysentimiento/robertuito-sentiment-analysis"
    )
except Exception as e:
    print(f"AVISO: Falha ao carregar modelo: {e}")
    analisador_sentimento_pt = None

#Função de analise de sentimento e retorno de polaridade (-1 a 1)
def analisar_sentimento(texto):
    if analisador_sentimento_pt is None:
        return "MODELO INDISPONÍVEL"
    
    resultado = analisador_sentimento_pt(texto)[0]
    label = resultado['label']
    score = resultado['score']
    
    if label == 'POS':
        return score
    elif label == 'NEG':
        return -score
    else: 
        return 0.0

# --- EXEMPLOS DE USO DO CÓDIGO ACIMA ---
textos = [
    "Eu adorei o atendimento do restaurante!",
    "O pedido demorou muito para chegar.",
    "A comida estava deliciosa e bem apresentada.",
    "Odiei. Nunca mais volto aqui."
]

for t in textos:
    palavras = extrair_palavras_chave(t)
    sentimento = analisar_sentimento(t)
    
    print(f"Texto: {t}")
    print(f"Palavras-chave: {palavras}")
    if isinstance(sentimento, str):
        print(f"Sentimento: {sentimento}\n")
    else:
        print(f"Sentimento: {sentimento:.4f}\n")
