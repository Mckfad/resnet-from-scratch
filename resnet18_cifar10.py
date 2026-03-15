# -*- coding: utf-8 -*-
"""
ResNet-18 from scratch - CIFAR-10
Yham Steeve Mackéols FADEGNON - ENEAM/ISE TP2

J'ai implémenté ça en relisant le papier ResNet (He et al., 2015).
L'astuce des skip connections, c'est pas si compliqué finalement :
c'est juste un "raccourci" qui garde l'information de l'entrée.

Pourquoi from scratch ? Parce qu'importer torchvision.resnet18()
c'est trop facile, je voulais vraiment comprendre comment ça marche
en pratique, surtout la partie où on additionne l'entrée avec la sortie.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import math, time, json, os

# Seed fixé - j'ai choisi 42 par habitude
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class WeightInitializer:
    """
    J'ai mis l'initialisation dans sa propre classe parce qu'au début
    je mélangeais tout dans le modèle et c'était le bazar.
    
    Pourquoi He et pas Xavier ? Parce que ReLU tue la moitié des neurones
    (ceux < 0), donc il faut compenser avec une variance ×2.
    Je l'ai appris à mes dépens après 3 échecs d'entraînement...
    """
    
    def __init__(self, strategy='he'):
        assert strategy in ('he', 'xavier')
        self.strategy = strategy
    
    def __call__(self, module):
        # Conv2d : initialisation des poids avec He ou Xavier
        if isinstance(module, nn.Conv2d):
            fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
            if self.strategy == 'he':
                std = math.sqrt(2.0 / fan_in)  # He : pour ReLU
            else:
                std = math.sqrt(1.0 / fan_in)  # Xavier : pour tanh/sigmoid
            nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Linear : même logique
        elif isinstance(module, nn.Linear):
            if self.strategy == 'he':
                std = math.sqrt(2.0 / module.in_features)
            else:
                std = math.sqrt(1.0 / module.in_features)
            nn.init.normal_(module.weight, 0.0, std)
            nn.init.zeros_(module.bias)
        
        # BatchNorm : gamma=1, beta=0 par défaut
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    @staticmethod
    def count_params(model):
        """Compte les paramètres - utile pour vérifier qu'on n'a pas de fuites"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


class ResidualBlock(nn.Module):
    """
    Le bloc de base de ResNet. J'ai galéré 2 jours sur le shortcut :
    quand on change de taille (stride=2), il faut projeter l'entrée aussi
    sinon on peut pas additionner (tailles différentes).
    
    Solution : Conv1x1 avec le même stride sur le shortcut.
    """
    
    def __init__(self, in_channels, out_channels, stride=1, use_batchnorm=True):
        super().__init__()
        
        # Chemin principal : 2 convolutions 3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=not use_batchnorm)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=not use_batchnorm)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
        
        # Le shortcut : l'astuce de ResNet
        self.shortcut = self._build_shortcut(in_channels, out_channels, stride, use_batchnorm)
    
    def _build_shortcut(self, in_ch, out_ch, stride, use_bn):
        """Construit le raccourci. J'avais un bug ici : oublié bias=False"""
        if stride == 1 and in_ch == out_ch:
            return nn.Identity()  # Cas simple : même taille, on copie
        
        # Cas compliqué : changement de taille ou de canaux
        # Il faut projeter avec Conv1x1 pour matcher les dimensions
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        identity = self.shortcut(x)  # On garde l'entrée (ou sa projection)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Addition résiduelle : le cœur de ResNet
        # Ça résout le vanishing gradient car ∂L/∂x = ∂L/∂out * (∂F/∂x + 1)
        # Le +1 garantit que le gradient ne meurt pas, même si F sature
        out = self.relu(out + identity)
        return out


class ResNet18(nn.Module):
    """
    Architecture complète. Adaptation pour CIFAR-10 :
    - Pas de MaxPool au début (contrairement à ImageNet) sinon 32x32 devient 4x4 trop vite
    - Stem en Conv 3x3 direct, pas de 7x7
    
    Architecture : [64, 128, 256, 512] canaux, strides [1, 2, 2, 2]
    """
    
    # Configuration : (canaux de sortie, nombre de blocs, stride)
    ARCH = [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]
    
    def __init__(self, num_classes=10, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        
        # Stem : entrée du réseau
        # Pourquoi pas MaxPool ? Parce que CIFAR c'est petit (32x32)
        # Si on met MaxPool comme dans ImageNet, on perd trop d'info
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        
        # Corps du réseau : empilement des blocs résiduels
        layers = []
        in_ch = 64
        for out_ch, n_blocks, stride in self.ARCH:
            layers.append(self._make_layer(in_ch, out_ch, n_blocks, stride))
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        
        # Tête de classification
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )
        
        # Initialisation He - essentiel, sinon ça converge pas
        self.apply(WeightInitializer(strategy='he'))
    
    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        """
        Crée une couche avec n_blocks.
        Le premier bloc gère le stride (éventuellement 2), 
        les suivants sont stride=1 (même taille).
        """
        blocks = [ResidualBlock(in_ch, out_ch, stride, self.use_batchnorm)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock(out_ch, out_ch, 1, self.use_batchnorm))
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.head(self.body(self.stem(x)))
    
    def summary(self):
        """Debug : affiche les shapes à chaque étape pour vérifier"""
        x = torch.randn(1, 3, 32, 32)
        print(f"ResNet-18 (use_batchnorm={self.use_batchnorm})")
        print("-" * 40)
        for name, module in [('stem', self.stem), ('body', self.body), ('head', self.head)]:
            x = module(x)
            print(f"{name:6} -> {tuple(x.shape)}")
        info = WeightInitializer.count_params(self)
        print("-" * 40)
        print(f"Paramètres: {info['total']:,} ({info['trainable']:,} entraînables)")


class CIFAR10DataModule:
    """
    Gestion des données. J'ai tout regroupé ici parce que j'en avais marre
    de recopier les mêmes transforms à chaque fois.
    
    Split : 90% train, 10% val (tirés du train original), 10k test séparé
    """
    
    # Stats CIFAR-10 (précalculées, pas besoin de les recalculer)
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
    CLASSES = ['avion', 'auto', 'oiseau', 'chat', 'cerf',
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    
    def __init__(self, batch_size=128, val_split=0.1, data_dir='./data', num_workers=2):
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def _train_transforms(self):
        """Data augmentation pour le train : crop aléatoire + flip horizontal"""
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Petit décalage aléatoire
            transforms.RandomHorizontalFlip(),      # Flip aléatoire (50%)
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
    
    def _test_transforms(self):
        """Pas d'augmentation pour val/test, juste la normalisation"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])
    
    def setup(self):
        """
        Télécharge et prépare les données.
        Premier appel = téléchargement (~170MB) dans ./data/
        """
        # Datasets
        train_full = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, download=True, 
            transform=self._train_transforms()
        )
        val_full = torchvision.datasets.CIFAR10(
            self.data_dir, train=True, download=True, 
            transform=self._test_transforms()
        )
        test_set = torchvision.datasets.CIFAR10(
            self.data_dir, train=False, download=True, 
            transform=self._test_transforms()
        )
        
        # Split train/val (90/10)
        n = len(train_full)
        indices = list(range(n))
        split = int(self.val_split * n)
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]
        
        # DataLoaders
        kwargs = {
            'batch_size': self.batch_size, 
            'num_workers': self.num_workers,
            'pin_memory': torch.cuda.is_available()
        }
        
        self.train_loader = DataLoader(
            train_full, sampler=SubsetRandomSampler(train_idx), **kwargs
        )
        self.val_loader = DataLoader(
            val_full, sampler=SubsetRandomSampler(val_idx), **kwargs
        )
        self.test_loader = DataLoader(test_set, shuffle=False, **kwargs)
        
        print(f"Données prêtes:")
        print(f"  Train: {len(train_idx):,} images ({len(self.train_loader)} batches)")
        print(f"  Val:   {len(val_idx):,} images ({len(self.val_loader)} batches)")
        print(f"  Test:  {len(test_set):,} images ({len(self.test_loader)} batches)")
    
    def denormalize(self, tensor):
        """Pour afficher les images : inverse la normalisation"""
        mean = torch.tensor(self.MEAN).view(3, 1, 1)
        std = torch.tensor(self.STD).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1)


class Trainer:
    """
    Boucle d'entraînement. J'ai choisi SGD+momentum (0.9) plutôt qu'Adam
    parce que pour ResNet sur CIFAR, SGD généralise mieux selon les papiers.
    
    C'est plus lent à converger au début, mais le résultat final est meilleur.
    Scheduler : lr * 0.1 aux epochs 100 et 150 (stratégie classique).
    """
    
    def __init__(self, model, device, lr=0.1, weight_decay=5e-4, save_path='best.pth'):
        self.model = model.to(device)
        self.device = device
        self.save_path = save_path
        
        self.criterion = nn.CrossEntropyLoss()
        
        # SGD avec momentum - meilleur que Adam pour ResNet/CIFAR
        self.optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
        
        # Decay du learning rate aux milestones
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, 
            milestones=[100, 150],  # Epochs 100 et 150
            gamma=0.1  # Division par 10
        )
        
        # Historique pour les courbes
        self.history = {
            'train_loss': [], 
            'val_loss': [], 
            'train_acc': [], 
            'val_acc': []
        }
        self.best_acc = 0.0
    
    def _run_epoch(self, loader, training=True):
        """
        Une epoch complète.
        J'ai séparé train et eval pour éviter d'oublier model.eval() ou train()
        """
        self.model.train(training)
        total_loss = correct = total = 0
        
        # Contexte : with_grad pour train, no_grad pour eval (économie mémoire)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if training:
                    self.optimizer.zero_grad()
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                if training:
                    # Backward et update
                    loss.backward()
                    self.optimizer.step()
                
                # Stats
                total_loss += loss.item() * len(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(images)
        
        return total_loss / total, 100.0 * correct / total
    
    def fit(self, train_loader, val_loader, epochs=200):
        """Entraînement complet avec sauvegarde du meilleur modèle"""
        print(f"Entraînement: {epochs} epochs | lr={self.optimizer.param_groups[0]['lr']}")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            # Val
            val_loss, val_acc = self._run_epoch(val_loader, training=False)
            
            # Update lr
            self.scheduler.step()
            
            # Logging
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Sauvegarde du meilleur modèle (sur val accuracy)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
            
            # Affichage : toutes les 10 epochs, ou chaque epoch si peu d'epochs
            if epochs <= 30 or epoch % 10 == 0 or epoch == 1:
                lr = self.scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                print(f"Ep {epoch:3d}/{epochs} | "
                      f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | "
                      f"lr={lr:.4f} | {elapsed:.0f}s")
        
        print(f"Meilleure val accuracy: {self.best_acc:.2f}%")
        return self.history
    
    def evaluate(self, loader):
        """Évaluation sur n'importe quel loader"""
        loss, acc = self._run_epoch(loader, training=False)
        return loss, acc
    
    def load_best(self):
        """Recharge le meilleur checkpoint trouvé pendant l'entraînement"""
        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device)
        )
        self.model.eval()


class AblationStudy:
    """
    Comparaison avec vs sans BatchNorm.
    
    Résultat attendu : sans BN, le réseau ne converge pas (reste à ~10% = aléatoire).
    Avec BN, on atteint ~80%. C'est l'expérience qui montre que BN est critique
    pour la stabilité de l'optimisation avec SGD et lr=0.1.
    """
    
    def __init__(self, data_module, device, epochs=200):
        self.dm = data_module
        self.device = device
        self.epochs = epochs
        self.results = {}
    
    def _run_one(self, use_batchnorm):
        """Entraîne une variante et retourne les résultats"""
        label = 'Avec BN' if use_batchnorm else 'Sans BN'
        savefile = 'resnet18_bn.pth' if use_batchnorm else 'resnet18_no_bn.pth'
        
        print(f"\n{'='*50}")
        print(f"Test: {label}")
        print('='*50)
        
        # Création modèle et entraînement
        model = ResNet18(num_classes=10, use_batchnorm=use_batchnorm)
        trainer = Trainer(model, self.device, lr=0.1, save_path=savefile)
        
        history = trainer.fit(self.dm.train_loader, self.dm.val_loader, self.epochs)
        
        # Évaluation finale sur test set
        trainer.load_best()
        _, test_acc = trainer.evaluate(self.dm.test_loader)
        
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return {
            'history': history,
            'test_acc': test_acc,
            'best_val': trainer.best_acc,
            'trainer': trainer
        }
    
    def run(self):
        """Lance les deux entraînements séquentiellement"""
        print("Lancement de l'ablation study...")
        print("Cela va entraîner 2 modèles : avec et sans BatchNorm")
        
        self.results['with_bn'] = self._run_one(True)
        self.results['without_bn'] = self._run_one(False)
    
    def print_summary(self):
        """Tableau comparatif final"""
        r_bn = self.results['with_bn']
        r_no = self.results['without_bn']
        
        print("\n" + "="*60)
        print("RÉSULTATS - Ablation Study (BatchNorm)")
        print("="*60)
        print(f"{'Métrique':<25} {'Avec BN':>10} {'Sans BN':>10} {'Diff':>10}")
        print("-"*60)
        
        metrics = [
            ('Val accuracy (best)', r_bn['best_val'], r_no['best_val']),
            ('Test accuracy', r_bn['test_acc'], r_no['test_acc']),
            ('Train acc (final)', r_bn['history']['train_acc'][-1], 
             r_no['history']['train_acc'][-1]),
        ]
        
        for name, v_bn, v_no in metrics:
            diff = v_bn - v_no
            print(f"{name:<25} {v_bn:>9.2f}% {v_no:>9.2f}% {diff:>+9.2f}%")
        
        print("-"*60)
        gain = r_bn['test_acc'] - r_no['test_acc']
        print(f"Gain apporté par BN: +{gain:.2f}%")
        print("="*60)
        
        if gain > 50:
            print("Conclusion: BatchNorm est CRITIQUE pour ce modèle.")
            print("Sans BN, SGD avec lr=0.1 diverge complètement.")
        else:
            print("Résultat surprenant - vérifier l'implémentation.")


# =============================================================================
# SCRIPT PRINCIPAL - Exécution directe
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ResNet-18 from scratch sur CIFAR-10'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Nombre d\'epochs (10=rapide pour tester, 200=complet)'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=128,
        help='Taille des batches'
    )
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Mode rapide: 10 epochs uniquement'
    )
    args = parser.parse_args()
    
    # Mode rapide (10) ou epochs personnalisés : python resnet18_cifar10.py --epochs 200 pour un entrainement complet par ex
    epochs = 10 if args.fast else args.epochs
    
    print("="*50)
    print("ResNet-18 From Scratch - CIFAR-10")
    print("="*50)
    
    # Chargement données
    print("\n[1/3] Préparation des données...")
    dm = CIFAR10DataModule(batch_size=args.batch_size)
    dm.setup()
    
    # Lancement ablation study
    print("\n[2/3] Entraînement (ablation study)...")
    study = AblationStudy(dm, device, epochs=epochs)
    study.run()
    
    # Résultats
    print("\n[3/3] Résultats:")
    study.print_summary()
    
    print("\n" + "="*50)
    print("Fichiers sauvegardés:")
    print("  - resnet18_bn.pth (modèle avec BatchNorm)")
    print("  - resnet18_no_bn.pth (modèle sans BatchNorm)")
    print("="*50)
