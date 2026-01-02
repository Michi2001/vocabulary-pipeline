import wn

# Einmalig das spanische WordNet laden
# wn.download("omw-es:1.4")

es_wn = wn.Wordnet("omw-es:1.4")
synsets = es_wn.synsets("zorro")

if len(synsets) == 0:
    print("Keine Synsets für 'zorro' gefunden.")

for syn in synsets:
    print(syn.id, syn.definition())
