PROYECTO: Sistema de Predicción Climática del Perú – NASA Space Apps Challenge 2025

 Introducción General

El Sistema de Predicción Climática del Perú, es una aplicación web impulsada por Inteligencia Artificial (IA) que combina modelos de aprendizaje automático, análisis de datos climáticos históricos y contexto geográfico para generar predicciones contextualizadas y recomendaciones sobre distintos sectores del país, como salud, economía y medio ambiente.

La aplicación fue desarrollada para el NASA Space Apps Challenge 2025, con el objetivo de crear una herramienta tecnológica que apoye la toma de decisiones informadas ante eventos climáticos extremos y fenómenos ambientales que afectan el desarrollo sostenible del Perú.

Objetivo del Proyecto

Diseñar y desarrollar una plataforma inteligente capaz de:
- Analizar datos climáticos históricos del Perú.
- Predecir comportamientos futuros (temperaturas, lluvias, fenómenos extremos).
- Relacionar estos resultados con indicadores sociales, económicos y ambientales.
- Ofrecer recomendaciones automáticas para las regiones del país según sus condiciones.

De esta forma, el sistema contribuye a la prevención de desastres, la planificación de recursos y la toma de decisiones basadas en datos.

Inteligencia Artificial y Datos Usados

El modelo fue entrenado con una base de datos de más de 416,000 registros climáticos en formato CSV, recopilados de distintas fuentes públicas peruanas (principalmente el SENAMHI), que abarcan el periodo 1990–2023.

Cada registro contiene variables como:
- Temperatura máxima y mínima
- Humedad
- Precipitación
- Altitud
- Región geográfica (Costa, Sierra, Selva)
- Fecha (mes, año)
- Fenómenos asociados (El Niño, sequías, heladas)

 Modelos de IA implementados:
Se entrenaron 6 modelos diferentes para comparar precisión y comportamiento:

1. Random Forest Regressor
2. Ridge Regression
3. Linear Regression
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

Después de la fase de entrenamiento, el modelo Random Forest fue seleccionado como el principal debido a su alta precisión promedio del 85%, especialmente en escenarios multivariables.

Arquitectura del Proyecto

El sistema está organizado en varias carpetas y archivos que trabajan de forma conjunta bajo el framework Flask:

📂 NasaChallenge/
├── app.py # Núcleo de la aplicación Flask
├── static/
│ ├── video/planet.mp4 # Fondo animado del planeta Tierra girando
│ ├── css/ # Estilos visuales y animaciones
│ └── js/ # Scripts y lógica de interfaz
├── templates/
│ ├── index.html # Página principal (interfaz de usuario)
│ └── results.html # Página de resultados y análisis
├── data/
│ └── registros_peru.csv # Datos históricos (416,000 registros)
└── README.md # Documentación del proyecto

Funcionamiento del Archivo `app.py`

El archivo `app.py` es el cerebro del sistema.  
Está desarrollado en Python e implementa el framework Flask, que permite integrar el modelo de inteligencia artificial con la interfaz web.

Funciones principales:
- Inicia el servidor local Flask.  
- Carga los modelos de IA previamente entrenados.  
- Recibe los datos ingresados por el usuario desde el formulario HTML.  
- Procesa esos datos (por ejemplo: ciudad, mes, variables climáticas).  
- Ejecuta la predicción utilizando el modelo correspondiente.  
- Devuelve el resultado acompañado de una descripción contextual según la región.

Datos Geográficos Incorporados

El archivo `app.py` también incluye un conjunto de datos predefinidos que representan las principales regiones del Perú con sus coordenadas y altitudes.  
Estos valores sirven para ajustar el contexto de las predicciones según la ubicación:

'data': {
  'lima': {'lat': -12.0464, 'lng': -77.0428, 'alt': 154},
  'cusco': {'lat': -13.5319, 'lng': -71.9675, 'alt': 3399},
  'arequipa': {'lat': -16.4090, 'lng': -71.5375, 'alt': 2335},
  'iquitos': {'lat': -3.7437, 'lng': -73.2516, 'alt': 106},
  'trujillo': {'lat': -8.1116, 'lng': -79.0290, 'alt': 34},
  'peru': {'lat': -9.1900, 'lng': -75.0152, 'alt': 500}
}

Con esta estructura, cada predicción no solo muestra valores numéricos, sino también una interpretación contextual, dependiendo de si la ciudad pertenece a la Costa, Sierra o Selva.

Interfaz Gráfica y Diseño Visual (HTML)
La interfaz fue diseñada para ser intuitiva y atractiva.
Se utilizó HTML5, CSS3 y JavaScript para crear un entorno visual moderno, acompañado por un video de fondo del planeta Tierra girando, simbolizando la conexión entre ciencia, datos y naturaleza.

Características visuales:
🎥 Fondo animado de un planeta girando (video .mp4)

🎛️ Panel central con formularios de entrada (selección de región, mes, año)

📊 Resultados presentados en tarjetas dinámicas y gráficos interactivos

🔔 Sección de alertas climáticas según nivel de riesgo

💬 Recomendaciones automáticas clasificadas por categoría:

🌱 Medio ambiente

💰 Economía

❤️ Salud
PROJECT: Peru's Climate Prediction System – NASA Space Apps Challenge 2025

General Introduction

The Peru's Climate Prediction System is a web application powered by Artificial Intelligence (AI) that combines machine learning models, historical climate data analysis, and geographic context to generate contextualized predictions and recommendations for various sectors of the country, such as health, the economy, and the environment.

The application was developed for the NASA Space Apps Challenge 2025, with the goal of creating a technological tool that supports informed decision-making in the face of extreme weather events and environmental phenomena that affect Peru's sustainable development.

Project Objective

To design and develop an intelligent platform capable of:
- Analyzing historical climate data from Peru.
- Predicting future behaviors (temperatures, rainfall, extreme events).
- Relating these results to social, economic, and environmental indicators.
- Offering automatic recommendations for the country's regions based on their conditions.

In this way, the system contributes to disaster prevention, resource planning, and data-driven decision-making.

Artificial Intelligence and Data Used

The model was trained with a database of more than 416,000 climate records in CSV format, compiled from various Peruvian public sources (mainly SENAMHI), covering the period 1990–2023.

Each record contains variables such as:
- Maximum and minimum temperature
- Humidity
- Precipitation
- Altitude
- Geographic region (Coast, Mountains, Jungle)
- Date (month, year)
- Associated phenomena (El Niño, droughts, frost)

AI models implemented:
Six different models were trained to compare accuracy and performance:

1. Random Forest Regressor
2. Ridge Regression
3. Linear Regression
4. Gradient Boosting
5. K-Nearest Neighbors (KNN)
6. Support Vector Machine (SVM)

After the training phase, the Random Forest model was selected as the primary model due to its high average accuracy of 85%, especially in multivariate scenarios.

Project Architecture

The system is organized into several folders and files that work together under the Flask framework:

📂 NasaChallenge/
├── app.py # Core of the Flask application
├── static/
│ ├── video/planet.mp4 # Animated background of the rotating planet Earth
│ ├── css/ # Visual styles and animations
│ └── js/ # Scripts and interface logic
├── templates/
│ ├── index.html # Main page (user interface)
│ └── results.html # Results and analysis page
├── data/
│ └── registro_peru.csv # Historical data (416,000 records)
└── README.md # Project documentation

How the `app.py` File Works

The `app.py` file is the brain of the system.
It is developed in Python and implements the Flask framework, which allows the artificial intelligence model to be integrated with the web interface.

Main Functions:
- Starts the local Flask server.
- Loads the pre-trained AI models.
- Receives the data entered by the user from the HTML form.
- Processes that data (e.g., city, month, weather variables).
- Runs the prediction using the corresponding model.
- Returns the result accompanied by a contextual description based on the region.

Integrated Geographic Data

The `app.py` file also includes a predefined dataset representing the main regions of Peru with their coordinates and altitudes.
These values ​​are used to adjust the context of the predictions based on the location:

'data': {
'lima': {'lat': -12.0464, 'lng': -77.0428, 'alt': 154},
'cusco': {'lat': -13.5319, 'lng': -71.9675, 'alt': 3399},
'arequipa': {'lat': -16.4090, 'lng': -71.5375, 'alt': 2335},
'iquitos': {'lat': -3.7437, 'lng': -73.2516, 'alt': 106},
'trujillo': {'lat': -8.1116, 'lng': -79.0290, 'alt': 34},
'peru': {'lat': -9.1900, 'lng': -75.0152, 'alt': 500}
}

With this structure, each prediction not only displays numerical values ​​but also a contextual interpretation, depending on whether the city belongs to the Coast, Mountains, or Jungle.

Graphical Interface and Visual Design (HTML)
The interface was designed to be intuitive and attractive.
HTML5, CSS3, and JavaScript were used to create a modern visual environment, accompanied by a background video of the rotating planet Earth, symbolizing the connection between science, data, and nature.

Visual Features:
🎥 Animated background of a rotating planet (.mp4 video)

🎛️ Central panel with input forms (select region, month, year)

📊 Results presented in dynamic cards and interactive graphics

🔔 Weather alerts section by risk level

💬 Automatic recommendations classified by category:

🌱 Environment

💰 Economy

❤️ Health
Example of a Real-World Forecast
Forecast for Lima – October 2027:

Expected temperature: 10.5°C
Severity level: HIGH (unusual event)
Model confidence: 85%
Estimated date: October 2027

Historical reference data:

Minimum: 8.2°C

Maximum: 34.2°C

Average: 19.0°C

The system automatically generates a contextual recommendation:

“Possible anomalous drop in temperature. Preventive measures are suggested for vulnerable sectors, especially public health and agriculture.”

Technologies Used
Area of ​​Implemented Technologies
Backend: Python, Flask, OS, Pathlib
AI/Data: Scikit-learn, Pandas, Numpy
Frontend: HTML5, CSS3, JavaScript
Visualization: Chart.js and custom components
Climate CSV Training (416,000 records)
Visual Design: Video background and interface effects

🧭 General System Flow
The user accesses the web interface (index.html).

They select a region (e.g., Cusco) and a month.

The form sends the request to app.py.

Flask executes the corresponding AI model.

The numerical result is obtained and contextualized with the geographic data.

The system displays the prediction and recommendations on the screen.

The results can be saved or reviewed later.

Development Team
Sergio Huaytan Oscategui
José Daniel Rubín Fonseca – Artificial Intelligence and Backend

Lenin Junior Sabino Diego
Haydenger Dany Paucar Silva – Web Development, Graphic Interface, and Visualization

(Collaborative climate innovation and analysis project for the NASA Space Apps Challenge 2025)

License
This project is licensed under the MIT License, which allows its free use, study, modification, and distribution for educational, scientific, or research purposes.

Conclusion
The Peruvian Climate Prediction System represents a technological proposal that combines artificial intelligence, open data, and interactive visualization to address one of the greatest current challenges: the impact of climate change.

Through the use of 416,000 real records and advanced predictive models, the system can anticipate climate behavior, generate early warnings, and offer useful information for sectors such as agriculture, health, the economy, and the environment.

Its integration of an immersive background video, its dynamic web architecture, and its solid scientific foundation make this project an innovative contribution to the challenge proposed by NASA.

🛰️ Project officially presented at the NASA Space Apps Challenge 2025
“Uniting data, science, and artificial intelligence for a sustainable future for Peru and the planet.”
