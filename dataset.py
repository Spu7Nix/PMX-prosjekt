# hvis du allerede har laget et datasett, vil denne filen vil prøve å skrive over det
# slett det forrige datasettet først for å få best mulig resultater

# biblioteker
import cutlet
import random
import json
import requests
import nltk
import os
import re
import string

# setup for japanske tegn -> latinske bokstaver
katsu = cutlet.Cutlet('kunrei')
katsu.use_foreign_spelling = False

# setup for setnings splitting
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# wikipedia API url
URL = "https://en.wikipedia.org/w/api.php"

# leser av en fil med en liste over wikipedia artikkelene som skal bli brukt
with open("articles.txt") as f:
    articles = f.readlines()

# tar vekk linjeskift fra artikkelnavnene
articles = [x.strip() for x in articles]

# språk som modellen skal lære
languages = [
    "en", "nb", "nn", "da", "sv", "es", "ja"
]

# lager datasettets filstruktur
main_dir_name = "./data"
if not os.path.exists(main_dir_name):
    os.mkdir(main_dir_name)
# en mappe for treningsdata, en mappe for testdata
for v in ["train", "test"]:
    if not os.path.exists(main_dir_name + "/" + v):
        os.mkdir(main_dir_name + "/" + v)
    # en mappe for hvert språk
    for lang in languages:
        if not os.path.exists(main_dir_name + "/" + v + "/" + lang):
            os.mkdir(main_dir_name + "/" + v + "/" + lang)

# initialiserer en liste for navnene til alle artikkelene
# på alle språkene
all_articles = []

# et nummer som blir brukt til å navngi hver fil
# blir også brukt til å sette hvilken data skal bli brukt til
# testing og hvilken til trening
index = 0

# hvor mye data som skal bli brukt til testing
test_data_size = 1000

for a in articles:
    # vi har allerede det engelske navnet
    all_articles.append(("en", a))

    # sender en request til wikipedia om titlen
    # på denne artikkelen på andre språk
    PARAMS = {
        "action": "query",
        "titles": a,
        "prop": "langlinks",
        "format": "json",
        "lllimit": 500,
    }
    print(f"skaffer språktitler for artikkel \"{a}\"")
    r = requests.get(url=URL, params=PARAMS).json()

    # legger dem til i listen
    for page in r["query"]["pages"].values():

        for lang in page["langlinks"]:
            if lang["lang"] in languages:
                all_articles.append((lang["lang"], lang["*"]))

# går gjennom alle artikkelene på alle språkene
for lang, name in all_articles:
    print(f"skaffer data fra artikkel \"{name}\" ({lang})")

    # laster ned et utdrag av artikkelen
    response = requests.get(
        f'https://{lang}.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format': 'json',
            'titles': name,
            'prop': 'extracts',
            'exintro': False,
            'explaintext': True,
        }
    ).json()
    page = next(iter(response['query']['pages'].values()))["extract"]

    # hvis språket er japansk, konverter tegnene til latinske bokstaver
    if lang == "ja":
        page = katsu.romaji(page).replace("?", "")

    # del artikkelen inn i setninger
    sentences = tokenizer.tokenize(page)

    for s in sentences:
        # fjern linjeskift
        t = s.replace('\n', ' ').lower()
        # fjern noen tall wikipedia setter inn for kilder eller noe
        t = re.sub(r"\[[0-9]*\]", "", t)
        # fjern tegnsetting
        t = t.translate(str.maketrans('', '', string.punctuation))

        # hvis setningen er mindre enn 10 bokstaver, fjern den
        if len(t) < 10:
            continue
        # hvis denne setningens index er mindre enn testdata størrelse
        # legg den til test datasettet, ellers legg det til train datasettet
        v = "test" if index < test_data_size else "train"

        # skriv setningen til datasettet
        file = open(f"{main_dir_name}/{v}/{lang}/{str(index)}.txt",
                    'w', encoding="utf-8")
        file.write(t)
        file.close()
        index += 1
