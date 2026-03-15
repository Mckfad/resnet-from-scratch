#!/usr/bin/env python3
"""
Test rapide d'une image - ResNet-18 CIFAR-10
Par Yham Steeve Mackéols FADEGNON

Bon, ce script c'est pour vous permettre de tester rapidement 
Vous avez un fichier .jpg ou .jpeg à tester? 
C'est ça que ce fichier fait.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os
from pathlib import Path

# J'importe ma classe ResNet18 du fichier principal
try:
    from resnet18_cifar10 import ResNet18
except ImportError:
    print("Euh... il me faut resnet18_cifar10.py dans le même dossier.")
    sys.exit(1)

# Les 10 classes de CIFAR-10, dans l'ordre du dataset
# J'ai gardé les noms en français comme dans mon code principal
CLASSES = ['avion', 'auto', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']


def charge_le_modele(checkpoint_path):
    """
    Bon alors voilà, je charge le modèle avec BatchNorm 
    parce que c'est la version qui marche. Sans BN ça converge pas,
    vous avez vu les résultats de mon ablation study.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(checkpoint_path):
        print(f"Hmm, je trouve pas {checkpoint_path}")
        print("Vérifiez qu'il est bien dans le dossier.")
        sys.exit(1)
    
    # Création du modèle avec BN (la version qui fonctionne)
    model = ResNet18(num_classes=10, use_batchnorm=True)
    
    # Chargement des poids
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()  # Mode évaluation, pas d'entraînement
    
    return model, device


def predit(model, image_path, device):
    """
    Là je fais la prédiction.
    - Je resize l'image à 32x32 (taille CIFAR)
    - Je normalise avec les stats du dataset
    - Je passe dans le modèle
    - Je renvoie la classe + les probabilités
    """
    # Les transformations CIFAR-10 classiques
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR c'est petit hein, 32x32 pixels
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],  # Moyennes CIFAR
                           [0.2470, 0.2435, 0.2616])   # Écarts-types
    ])
    
    # J'ouvre l'image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Problème avec l'image : {e}")
        sys.exit(1)
    
    # Je prépare le tensor
    tensor = transform(img).unsqueeze(0).to(device)  # unsqueeze = batch de 1
    
    # Prédiction sans calcul de gradients (plus rapide, moins de mémoire)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confiance, prediction = probs.max(1)
    
    # Je récupère le top 3 pour info
    top3_conf, top3_idx = torch.topk(probs, 3, dim=1)
    top3 = [(CLASSES[i], float(p)) 
            for i, p in zip(top3_idx.squeeze().cpu().numpy(), 
                           top3_conf.squeeze().cpu().numpy())]
    
    return CLASSES[prediction.item()], float(confiance.item()), top3


def main():
    # Vérification des arguments 
    if len(sys.argv) < 2:
        print("Usage : python test_one_image.py votre_image.jpg")
        print("        python test_one_image.py votre_image.jpg mon_checkpoint.pth")
        print("")
        print("Si vous précisez pas le checkpoint, j'utilise resnet18_bn.pth par défaut.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else "resnet18_bn.pth"
    
    # Vérification que l'image existe
    if not os.path.exists(image_path):
        print(f"L'image {image_path} existe pas là.")
        sys.exit(1)
    
    # Affichage des infos
    print(f"\n🖼️  Image à tester : {Path(image_path).name}")
    print(f"📦 Checkpoint utilisé : {checkpoint}")
    
    # Chargement
    print("\nJe charge le modèle...", end=" ")
    model, device = charge_le_modele(checkpoint)
    print(f"OK (sur {device})")
    
    # Prédiction
    print("J'analyse l'image...", end=" ")
    classe_predite, confiance, top3 = predit(model, image_path, device)
    print("OK")
    
    # Affichage du résultat
    print(f"\n{'='*50}")
    print(f"🎯 RÉSULTAT : {classe_predite.upper()}")
    print(f"   Confiance : {confiance:.1%}")
    print(f"{'='*50}")
    
    print(f"\n📊 Top 3 des prédictions :")
    for i, (nom_classe, proba) in enumerate(top3):
        # Petite barre de progression visuelle
        barre = "█" * int(proba * 20)
        fleche = "→" if i == 0 else " "
        print(f"   {fleche} {nom_classe:>10} : {proba:>6.1%} {barre}")
    
    print("")  


if __name__ == "__main__":
    main()
