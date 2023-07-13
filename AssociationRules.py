import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Charger la base de données Zoo
zoo_data = pd.read_csv('zoo/zoo.csv')

# Extraire les colonnes nécessaires pour l'analyse des règles d'association
association_data = zoo_data.drop('animal_name', axis=1)

# Convertir les données en codage One-Hot
association_data = pd.get_dummies(association_data)
association_data = association_data.astype(bool)

# Extraire les itemsets fréquents avec un support minimum de 0.1
frequent_itemsets = apriori(association_data, min_support=0.1, use_colnames=True)

# Générer les règles d'association avec une confiance minimale de 0.9
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.9)

# Sélectionner les règles non redondantes
if 'redundant' in rules.columns:
    non_redundant_rules = rules[~rules['redundant']]
else:
    non_redundant_rules = rules

# Visualiser les règles non redondantes
print(non_redundant_rules)
non_redundant_rules.to_excel('zoo/non_redundant_rules.xlsx', index=False)
