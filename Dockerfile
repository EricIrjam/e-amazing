# Utiliser une image de base officielle de Python 3.10
FROM python:3.10

# Définir un répertoire de travail dans le conteneur pour votre application
WORKDIR /app

# Copier le fichier .parquet dans le conteneur à l'emplacement approprié
COPY ./data/full_df_output.parquet /app/data/full_df_output.parquet

# Copier le script Python principal dans le conteneur
COPY ./5_RMF_full_data.py /app

# Copier le fichier requirements.txt pour gérer les dépendances Python
COPY ./requirements.txt .

# Installer les dépendances Python spécifiées dans requirements.txt sans utiliser le cache
RUN pip install --no-cache-dir -r requirements.txt

# Spécifier la commande à exécuter au démarrage du conteneur : exécuter le script Python
CMD ["python", "5_RMF_full_data.py"]
