# Gerador de frases
# Linguagens Formais e Autômatos LAB - 2020
# 160027 - Gabriel Yudi Sanefugi
# 150017 - Lauren Maria Ferreira
# 000000 - Nathalia Lourenço

# As regras do gerador representam uma construção simples de uma frase em inglês. A frase é composta por:
# <artigo> <substantivo> <verbo conjugado> <adjetivo>
# O algoritmo seleciona um substantivo aleatoriamente, verifica a conjugação do verbo (se é plural ou singular, 
# que é um atributo booleano "plural" do objeto substantivo) e seleciona um adjetivo aleatório para gerar a sentença.
# Não foi feita uma curadoria detalhada das frases geradas, então semanticamente muitas frases podem não fazer sentido.
# O projeto será evoluído no decorrer do curso.

import json
from random import randint

# abertura do arquivo JSON, armazenamento em variável e fechamento do arquivo JSON
with open('../input/gerador-de-frases/adjectives.json') as f:
        adjetivos = json.load(f)
with open('../input/gerador-de-frases/noun.json') as f:
        substantivos = json.load(f)

# escolha aleatória do substantivo
substantivo = substantivos['nouns'][randint(0,len(substantivos['nouns']))]

# conjugação (singular/plural [is/are])
if substantivo['plural']:
    conjugacao = 'are'
else:
    conjugacao = 'is'

#escolha aleatória do adjetivo
adjetivo = adjetivos['adjectives'][randint(0,len(adjetivos['adjectives']))]

# forma a frase
print("The "+ substantivo["noun"] + " " + conjugacao + " " + adjetivo)