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

---

## 📊 Modèle Utilisé

- Modèle : CNN Séquentiel (Keras)  
- Accuracy sur données propres : **92.5 %**

---
<center><img width="512" height="852" alt="interfac web complet" src="https://github.com/user-attachments/assets/1ae09678-5b87-45f3-b428-c48a89d06948" /></center>


## 👩‍💻 Auteur

**Nour Elwoujoud Khémiri**  
📧 khemirinour334@gmail.com  
📍 Faculté des Sciences de Sfax — Mastère professionnel en Cybersécurité et Industrie Intelligente  
🔗 [LinkedIn](https://www.linkedin.com/in/nour-elwoujoud-khemiri-0463a3209/)
