# Prueba Técnica: Data Scientist

## 1 - Descripción

Los datos provienen del proyecto [Inside Airbnb](http://insideairbnb.com/about.html) y describen el uso de la pltaforma de alquileres temporarios Airbnb en la ciudad de Madrid. Están compuestos de 7 archivos:

Archivo | Descripción
--------|------------
[listings.csv.gz](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/data/listings.csv.gz) | Detalle de las publicaciones
[calendar.csv.gz](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/data/calendar.csv.gz) | Datos detallados del calendario para las publicaciones
[reviews.csv.gz](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/data/reviews.csv.gz) | Datos detallados de las reseñas
[listings.csv](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/visualisations/listings.csv) | Resumen de información y métricas para las publicaciones
[reviews.csv](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/visualisations/reviews.csv) | Resumen de reseñas 
[neighbourhoods.csv](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2020-01-10/visualisations/neighbourhoods.csv) | Lista de barrios. Procedente de archivos GIS de código abierto o de la ciudad
[neighbourhoods.geojson](http://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/2023-03-15/visualisations/neighbourhoods.geojson) | Archivo GeoJSON de barrios de la ciudad

Se plantea un escenario hipotético en el que un cliente posee propiedades disponibles para su alquiler, distribuidas en diferentes barrios de la ciudad. Además de un análsis detallado de la situación, el cliente quiere poder predecir el precio final al que podría alquilar cada propiedad. 

## 2 - Objetivos específicos 

### 2.1 - Analisis exploratorio

*  2.1.1 - Describir la situación del mercado de alquileres temporales en Airbnb en la ciudad de Madrid a nivel general. Se valorarán las descripciones visuales y la explotación de la riqueza de los datos. Por ejemplo, haciendo uso de los datos geoespaciales y/o del lenguaje natural.

*  2.1.2 - Analizar los precios de las propiedades publicadas.

### 2.2 - Predicción

* 2.2.1 - Entrenar un modelo capaz de predecir el precio de alquiler diario de una propiedad en Airbnb. Además de las métricas obtenidas, se valorará la justificación del proceso de construcción del modelo (por ejemplo: variables utilizadas/descartadas, métrica/s de evaluación, selección de modelo/s, etc.), la creatividad en la construcción de nuevas variables (utilizando los datos de geolocalización y/o texto no estructurado) y el uso de diferentes técnicas predictivas.

## 3 - Entregables

* Código (Puedes utilizar R y/o Python)
* Documentación
* Informe (y/o presentación) con los resultados del análisis y la interpretación de las métricas de predicción obtenidas. Se valorarán positivamente las recomendaciones de negocio.
* Propuesta de despliegue y/o productivización de la solución (opcional)

## 4 - Extras

Si quieres/puedes hacer algún otro tipo de demostración de habilidades técnicas, aprovechando la riqueza de los datos, anímate! Lo tendremos en cuenta en la evaluación.


# Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── pyproject.toml   <- The requirements file for reproducing the analysis environment with poetry
    │
    ├── poetry.lock   <- Poetry lock file for specific requirement versions  
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

