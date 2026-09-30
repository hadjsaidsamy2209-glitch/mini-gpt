# Mini-GPT from scratch

Un petit modèle de langage de type GPT, codé **de zéro** en Python avec PyTorch, en suivant la vidéo de Andrej Karpathy
[« Let's build GPT: from scratch, in code, spelled out »](https://www.youtube.com/watch?v=kCc8FmEb1nY).

Le modèle apprend à **prédire le caractère suivant** à partir du texte qui précède. Entraîné sur du Shakespeare, il génère du texte
qui respecte la structure d'une pièce de théâtre (noms de personnages en majuscules, dialogues, ponctuation).

## Ce que le projet contient

- Tokenisation au niveau des caractères (encodeur et décodeur)
- Découpage des données en jeu d'entraînement (90 %) et de validation (10 %)
- **Self-attention** avec masque causal (le modèle ne voit pas le futur)
- **Multi-head attention** (plusieurs têtes en parallèle)
- Couche **feed-forward** (Linear + ReLU)
- **Blocs Transformer** : connexions résiduelles, LayerNorm (pre-norm), dropout
- Plongements de position (le modèle connaît l'ordre des caractères)
- Boucle d'entraînement (AdamW), suivi de la loss d'entraînement et de validation
- Génération de texte et sauvegarde des poids (`mini_gpt.pth`)

## Hyperparamètres utilisés

Version réduite, adaptée à un GPU de 4 Go (GTX 1650) :

| Paramètre | Valeur |
|---|---|
| `batch_size` | 32 |
| `block_size` (contexte) | 128 |
| `n_embd` | 192 |
| `n_head` | 6 |
| `n_layer` | 4 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 |
| `max_iters` | 5000 |

Nombre de paramètres : environ **1,8 million**.

## Résultats

| Étape | Loss d'entraînement | Loss de validation |
|---|---|---|
| 0 | 4.32 | 4.32 |
| 500 | 2.34 | 2.35 |
| 2000 | 1.70 | 1.87 |
| 4999 | 1.44 | **1.63** |

La loss de départ (environ 4.3) correspond à un choix au hasard parmi 65 caractères. Avec une loss de validation de 1.63,
le modèle hésite en moyenne entre environ 5 caractères à chaque position.

Extrait de texte généré (après 5000 étapes) :

```
GLOUCESTER:
To will,never that takes as for her Glorge, more;
For with Peard HaRf's mustages.

ELBOW:
No made, and good
villyfance, beards lagard away from to now...
```

Le modèle a appris la **forme** du texte, pas son **sens** : il imite le style mais ne comprend rien.

## Lancer le projet

1. Installer les dépendances :

```bash
pip install torch
```

Pour utiliser un GPU NVIDIA, installer la version de PyTorch qui correspond à CUDA (voir pytorch.org).

2. Télécharger le texte d'entraînement (Tiny Shakespeare, environ 1 Mo) et le nommer `input.txt` :

```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')"
```

3. Lancer l'entraînement et la génération :

```bash
python biagram.py
```

La loss s'affiche toutes les 500 étapes. À la fin, le script génère 500 caractères et enregistre le modèle dans `mini_gpt.pth`.

## Limites

- Modèle minuscule (1,8 M de paramètres, contre des milliards pour les vrais LLM) entraîné sur environ 1 Mo de texte.
- Modélisation caractère par caractère : pas de tokens de mots.
- C'est un projet d'apprentissage : il sert à comprendre le mécanisme de ChatGPT (attention, blocs Transformer, entraînement),
  pas à produire un assistant utilisable.

## Pistes d'amélioration

- Utiliser les hyperparamètres de la vidéo (`n_embd=384`, 6 couches, `block_size=256`) sur un GPU plus puissant (Google Colab), pour viser une loss d'environ 1.5.
- Entraîner sur un autre corpus (textes français libres de droits, par exemple).
- Passer à un tokenizer par sous-mots (BPE).
- Charger un modèle déjà entraîné en local (Ollama) et l'utiliser avec mes propres documents (RAG).

## Crédits

Projet réalisé en suivant le cours de Andrej Karpathy (vidéo et dépôt `ng-video-lecture`). Le code a été écrit et débogué pas à pas
pendant l'apprentissage.
