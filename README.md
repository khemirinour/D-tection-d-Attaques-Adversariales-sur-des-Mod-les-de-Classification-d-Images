# 🛡️ Détection d'Attaques Adversariales pour les modèles de classification d'images


## 📌 Description du Projet
Ce projet vise à développer un pipeline complet pour générer, détecter et se défendre contre les attaques adversariales sur des modèles de classification d'images. Il est particulièrement adapté aux cas d’usage en imagerie médicale (IRM), où la sécurité et la fiabilité des modèles sont cruciales.

---

## 🧠 Objectifs

- Implémenter des attaques adversariales populaires : FGSM, PGD, BIM, MIM, Carlini & Wagner  
- Tester la robustesse des modèles de classification  
- Intégrer des techniques de défense : adversarial training, preprocessing, détection  
- Déployer une API Flask pour automatiser le pipeline  
- Intégrer une interface web pour l’expérimentation utilisateur  

---

## 🧪 Exemples d'Attaques Implémentées

### 🔹 FGSM (Fast Gradient Sign Method)

**Définition :**  
Le FGSM est une attaque à un seul pas qui perturbe légèrement l’image originale dans la direction du gradient du modèle pour maximiser la perte.

**Équation (format texte) :**  
`x_adv = x + ε * sign(∇_x J(θ, x, y))`

---

### 🔹 PGD (Projected Gradient Descent)

**Définition :**  
Le PGD est une version itérative et plus puissante de FGSM, qui applique plusieurs perturbations tout en projetant l’image modifiée dans un voisinage autorisé.

**Équation (format texte) :**  
`x_adv(t+1) = Projection_epsilon(x_adv(t) + α * sign(∇_x J(θ, x_adv(t), y)))`

---

### 🔹 BIM (Basic Iterative Method)

**Définition :**  
Le BIM est similaire au PGD mais sans projection explicite. Il applique plusieurs petites perturbations successives.

**Équation (format texte) :**  
`x_adv(t+1) = Clip_{x, ε}(x_adv(t) + α * sign(∇_x J(θ, x_adv(t), y)))`

---

### 🔹 MIM (Momentum Iterative Method)

**Définition :**  
Le MIM introduit un terme de momentum dans l’attaque itérative, ce qui permet de stabiliser la direction du gradient et d’améliorer la transférabilité de l’attaque vers d’autres modèles.

**Équation (format texte) :**  
```
g(t+1) = μ * g(t) + ∇_x J(θ, x_adv(t), y) / L1_norm(∇_x J(θ, x_adv(t), y))
x_adv(t+1) = Clip_{x, ε}(x_adv(t) + α * sign(g(t+1)))
```

---

### 🔹 Carlini & Wagner (C&W)

**Définition :**  
L’attaque C&W repose sur une optimisation qui minimise la perturbation tout en s’assurant que l’image est mal classée.

**Équation (format texte) :**  
```
minimize ||δ||² + c * f(x + δ)
subject to x + δ ∈ [0,1]^n
```
## 🏗️ Architecture du Pipeline

Voici l’architecture globale du pipeline utilisé pour la détection d’attaques adversariales :

<img width="851" height="772" alt="pipline" src="https://github.com/user-attachments/assets/ddf8bdbc-a15e-4de4-9b3d-6be1ea60fc8a" />


Afin de pallier l’insuffisance de jeux de données publics contenant des exemples d’images médicales adversariales, nous avons généré notre **propre dataset adversarial** à partir d’images IRM d’origine. Celui-ci a été utilisé pour l’entraînement et la validation de notre module de détection d’attaques adversariales.

Les étapes du pipeline sont les suivantes :

1. **Prétraitement des images** : redimensionnement, normalisation.  
2. **Génération d’attaques adversariales** : implémentation de FGSM, PGD, BIM, MIM, C&W.  
3. **Création du dataset adversarial** : sauvegarde des images perturbées avec étiquettes.  
4. **Entraînement du détecteur** : apprentissage supervisé sur des paires (image propre, image attaquée).  
5. **Détection conditionnelle** :  
   - Si l’image est détectée **propre**, elle est envoyée au **classificateur principal** pour prédiction de la classe.  
   - Si l’image est détectée **adversariale**, elle est signalée ou rejetée pour éviter une mauvaise classification.  
6. **Déploiement via API Flask** : automatisation du pipeline.  
7. **Interface web (optionnelle)** :  
   - Permet d’**importer** le dataset d’images propres ainsi que le modèle de classification, afin de générer un dataset adversarial personnalisé et d’entraîner un détecteur adapté.  
   - Offre une **option de test en temps réel** :  
     - L’utilisateur peut uploader une image pour analyse immédiate.  
     - Le système utilise le détecteur pour déterminer si l’image est propre ou adversariale.  
     - Si l’image est **propre**, elle est automatiquement classifiée par le modèle principal.  
     - Si l’image est **adversariale**, le système affiche le type d’attaque détecté, permettant ainsi une meilleure interprétation et gestion.

---

## 📊 Modèle Utilisé

- Modèle : CNN Séquentiel 
- Accuracy sur données propres : **92.5 %**

---
<img width="512" height="852" alt="interfac web complet" src="https://github.com/user-attachments/assets/1ae09678-5b87-45f3-b428-c48a89d06948" />
<img width="736" height="1131" alt="predict = propre" src="https://github.com/user-attachments/assets/ce556a39-bffb-4b68-a601-3ef2e546d0bd" />


## 👩‍💻 Auteur

**Nour Elwoujoud Khémiri**  
📧 khemirinour334@gmail.com  
📍 Faculté des Sciences de Sfax — Mastère professionnel en Cybersécurité et Industrie Intelligente  
🔗 [LinkedIn](https://www.linkedin.com/in/nour-elwoujoud-khemiri-0463a3209/)
