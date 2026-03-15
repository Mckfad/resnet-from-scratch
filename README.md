
# ResNet-18 From Scratch — CIFAR-10

**Par :** Yham Steeve Mackéols FADEGNON  
**Cours :** Advanced Deep Learning — ENEAM/ISE TP2

---

## En résumé

J'ai implémenté ResNet-18 from scratch sur CIFAR-10. Pas de `torchvision.models.resnet18()`, j'ai tout codé moi-même pour vraiment comprendre comment les skip connections fonctionnent. Résultat : ~80% de précision avec BatchNorm, ~10% sans (l'ablation study montre bien que BN est critique).

---

## Structure du dépôt

```
resnet-from-scratch/
├── resnet18_cifar10.py      # Code principal (toutes les classes)
├── test_one_image.py        # Pour tester une image rapidement
├── requirements.txt         # Dépendances
├── README.md               # Ce fichier
└── resnet18_bn.pth          # Modèle entraîné (42.7 MB, voir Releases)
```

**Checkpoints :**
- `resnet18_bn.pth` — Modèle avec BatchNorm (**42.7 MB**, SHA256: `3a82ed223e2eb8433e506e2612c909f3f4b5f1e92d136b219a535dd14762c548`)
- Disponible dans [GitHub Releases](../../releases) (fichier trop gros pour git)

---

## Installation rapide

```bash
# Clone
git clone https://github.com/[votre-username]/resnet-from-scratch.git
cd resnet-from-scratch

# Dépendances
pip install torch torchvision pillow

# Téléchargez le modèle depuis GitHub Releases
# Placez resnet18_bn.pth dans le dossier
```

---

## Utilisation

### 1. Tester une image (pour l'évaluation rapide)

Vous avez une image et vous voulez voir si mon modèle la reconnaît :

```bash
python test_one_image.py votre_image.jpg
```

**Exemple de sortie :**
```
🖼️  Image à tester : chat.png
📦 Checkpoint utilisé : resnet18_bn.pth

Je charge le modèle... OK (sur cuda)
J'analyse l'image... OK

==================================================
🎯 RÉSULTAT : CHAT
   Confiance : 94.2%
==================================================

📊 Top 3 des prédictions :
   →       chat :  94.2% ████████████████████
      chien :   3.1% █▌
    oiseau :   1.2% ▌
```

Si vous voulez utiliser un checkpoint spécifique :
```bash
python test_one_image.py mon_image.jpg mon_checkpoint.pth
```

### 2. Réentraîner le modèle

```bash
# Rapide (10 epochs, ~4 min sur GPU) — pour tester que ça tourne
python resnet18_cifar10.py --epochs 10

# Complet (200 epochs, ~25 min sur GPU) — pour reproduire mes résultats
python resnet18_cifar10.py --epochs 200
```

Ça va lancer l'ablation study automatiquement (avec BN puis sans BN) et afficher le tableau comparatif à la fin.

---

## Architecture

J'ai adapté ResNet-18 pour CIFAR-10 (32×32 pixels) :

```
Input (3×32×32)
    ↓
Conv3×3, 64 → BN → ReLU          [Stem : pas de MaxPool pour CIFAR]
    ↓
Layer 1 : 2× ResBlock(64→64)       [stride 1, 32×32]
    ↓
Layer 2 : 2× ResBlock(64→128)      [stride 2, 16×16]
    ↓
Layer 3 : 2× ResBlock(128→256)     [stride 2, 8×8]
    ↓
Layer 4 : 2× ResBlock(256→512)     [stride 2, 4×4]
    ↓
GlobalAvgPool → Flatten → Linear(512→10)
```

**Pourquoi pas de MaxPool au début ?**  
Sur ImageNet on met Conv7×7 + MaxPool pour réduire rapidement. Mais CIFAR c'est déjà petit (32×32), si je fais ça j'arrive trop vite à 4×4 et j'perds trop d'info. Donc je commence direct avec Conv3×3.

---

## Ce qui fait la différence

| Choix technique | Pourquoi |
|-----------------|----------|
| **He initialization** (`std = √(2/fan_in)`) | ReLU tue 50% des neurones, faut compenser. J'ai testé Xavier, ça converge moins bien. |
| **SGD + momentum 0.9** | Adam converge plus vite au début mais généralise moins bien sur CIFAR. |
| **MultiStepLR [100,150]** | Division par 10 du LR aux epochs 100 et 150. Classique pour CIFAR. |
| **BatchNorm** | Sans ça, SGD avec lr=0.1 explose complètement. L'ablation montre +70% de gain. |

---

## Résultats de l'ablation study

| Métrique | Avec BN | Sans BN | Différence |
|----------|---------|---------|------------|
| Val accuracy (meilleure) | ~80% | ~10% | **+70%** |
| Test accuracy | ~80% | ~10% | **+70%** |
| Train accuracy (finale) | ~82% | ~10% | **+72%** |

**Conclusion :** BatchNorm est absolument critique. Sans lui, le réseau reste à 10% (aléatoire) même après 200 epochs.

---

## Organisation du code (OOP)

J'ai séparé en classes pour que ce soit propre et testable :

| Classe | Rôle |
|--------|------|
| `WeightInitializer` | Initialisation He/Xavier, appliquée avec `model.apply()` |
| `ResidualBlock` | Le bloc de base : 2 Conv3×3 + skip connection |
| `ResNet18` | L'architecture complète (11.17M paramètres) |
| `CIFAR10DataModule` | Chargement, augmentation, split train/val/test |
| `Trainer` | Boucle d'entraînement avec scheduler et checkpointing |
| `AblationStudy` | Lance les deux entraînements et compare |

---

## FAQ

**Q : Comment je gère le vanishing gradient ?**  
A : Les skip connections. Mathématiquement, ∂L/∂x = ∂L/∂out · (∂F/∂x + 1). Le +1 garantit que le gradient ne s'effondre pas, même dans les couches profondes.

**Q : Pourquoi He et pas Xavier ?**  
A : Xavier c'est pour tanh/sigmoid (fonctions symétriques). ReLU c'est asymétrique (tout à 0 pour x<0), donc il faut une variance ×2 pour compenser. Formule : `std = √(2/fan_in)` au lieu de `√(1/fan_in)`.

**Q : Pourquoi pas de MaxPool initial ?**  
A : CIFAR-10 c'est 32×32. Avec MaxPool je descends trop vite à 4×4 et je perds l'information spatiale. ImageNet c'est 224×224, là on peut se le permettre.

**Q : Pourquoi SGD et pas Adam ?**  
A : Pour ResNet sur CIFAR, SGD+momentum donne une meilleure généralisation finale. Adam converge plus vite au début mais plateau plus tôt. C'est un choix classique en vision.
