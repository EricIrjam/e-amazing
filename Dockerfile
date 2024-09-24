# Utiliser une image de base avec Java et Python préinstallés
FROM openjdk:11-jdk

# Installer les dépendances système nécessaires pour construire certains paquets Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    gfortran \
    libpng-dev \
    libfreetype6-dev \
    && apt-get clean

# Mettre à jour pip, setuptools et wheel pour assurer la compatibilité
RUN pip install --upgrade pip setuptools wheel

# Installer PySpark, Jupyter Notebook, scikit-learn, pandas, matplotlib, seaborn, pyarrow, fastparquet, dask et TensorFlow 2.13.0
RUN pip install --no-cache-dir --timeout=100 \
    pyspark \
    jupyter \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    pyarrow \
    fastparquet \
    dask[complete] \
    tensorflow==2.13.0 \
    tensorflow-addons

# Rétrograder NumPy si nécessaire pour la compatibilité avec TensorFlow
RUN pip install --no-cache-dir numpy==1.23.5

# Créer un dossier de travail dans le conteneur
WORKDIR /app

# Copier le fichier .parquet dans le conteneur
# Assurez-vous que le fichier full_df_output.parquet est dans le même dossier que votre Dockerfile
COPY ./data/full_df_output.parquet /app/data/full_df_output.parquet

# Copier le reste du contenu dans le conteneur
COPY . /app

# Exposer le port pour Jupyter Notebook
EXPOSE 8888

# Commande par défaut pour démarrer Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
