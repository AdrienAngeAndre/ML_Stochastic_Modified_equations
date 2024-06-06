# ML_Stochastic_Modified_equations

Ce code à pour but en partant d'une EDS simple et d'un intégrateur numérique de créer un réseau de neurones pour l'apprentissage d'une EDS modifié, on retouve dans ce dossier plusieurs éléments : 

slide_EDS.pdf corresponds aux slides de présentations du travail lors d'une soutenance 

Rapport.pdf corresponds au raport illustrant la problématique ainsi que le travail effectué de manière détaillé 

Figure contient les figures utilisés dans le rapport et la présentation     

Bibliography contient les références bibliographiques du travail 

NN_EDO.py contient ma première version du travail dans le cas non stochastique 

code contient le code finale pour la version stochastique du travail, avec plus de détails : 

code/Field.py contient la définition d'un objet représentant un champs de vecteur ainsi que 2 extensions de cette classe représentant un champs de vecteur modifié par Machine Learning et analytiquement. On y retrouve aussi des fonctions illustrant des champs de vecteurs classiques. Les seuls champs de vecteur implémenter pour l'instant sont les champs de vecteurs utilisés pour l'EDS Linéaire et pour le Pendule Stochastique.

code/Schéma.py contient la définition d'un objet représentant un intégrateur numérique, il prends en paramètre le type d'integrateur ainsi que les 2 champs de vecteurs de l'EDS associé (qui peuvent être les même / qui peuvent êtres modifiés). Les types d'intégrateurs implémentés pour l'instant sont Euler Maruyama / Point Milieu ainsi qu'une version réecrite de EMaruyama pour s'adpater au cas Linéaire. Il contient aussi des fonctions auxiliaires aux Schéma numériques comme un Point Fixe pour MidPoint 

code/models.py contient tout ce qui attrait au réseau de neurones cad la défintion des classes de réesaux comme les MLP, la fonction créatrice des modèles en fonctions du problème et les fonctions ayant attrait à l'apprentissage de modèle comme la fonction pour entrainer un lot ou pour calculer la perte.

code/Init.py contient les fonctions d'initialisations des données, on y retrouve les fonctions créants les générateurs de N quadruplets (y0,h,E(y1),V(y1)) servant de données d'entrainement ou de tests. On y retrouve aussi la fonction créant l'optimiser. 

code/Plot.py contient les fonctions relatives à l'affichage des metrics importantes de problèmes comme l'évolution de la perte, l'erreur faible... 

code/Variables.py est un fichier de commande dans lequel on retrouve les données relatives aux réseaux de neurones comme les hyperparamètres. On y retrouve aussi la definition d'un obejct stockant les variables de notre système numérique comme le type d'intégrateur ou les types des termes de drift ou de diffusions qui servent à intialiser les objets présents dans les autres fichiers. On y retouve aussi des valeurs arbitraires de ces paramètres utilisés dans le fichier NN_EDS.py    

code/NN_EDS.py est le fichier principal, il execute un entrainement et plot la Loss en utilisant les paramètres présents dans le fichier Variables

code/Linear.py est un fichier de test, il execute un entrainement et plot la Loss et d'autres metrics comme l'erreur faible en utilisant des paramètres correspondant au test effectué pour le système Linéaire codés en durs dans le fichier

code/Pendule.py est un fichier de test, il execute un entrainement et plot la Loss et d'autres metrics comme l'erreur faible en utilisant des paramètres correspondant au test effectué pour le système Pendule stochastique codés en durs dans le fichier 

code/Additif.py est un fichier de test, il execute un entrainement et plot la Loss et d'autres metrics comme l'erreur faible en utilisant des paramètres correspondant au test effectué pour le système Additif codés en durs dans le fichier  
