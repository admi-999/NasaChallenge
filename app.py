#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import json
import io
from datetime import datetime
from pathlib import Path
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB máximo

class AdvancedClimatePredictor:
    """Predictor de clima avanzado para el dashboard"""
    
    def __init__(self, model_dir='trained_models_full'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.poly_features = {}
        self.model_scores = {}
        self.feature_columns = []
        self.historical_data = {}
        self.load_models()
        self.load_historical_data()

    def load_models(self):
        """Carga modelos entrenados con manejo robusto de errores"""
        try:
            # Buscar directorio de modelos
            possible_dirs = [
                self.model_dir,
                'trained_models_full',
                '../trained_models_full',
                'trained_models_partial',
                '../trained_models_partial',
                'models',
                '../models',
                'backend/models',
                'backend/trained_models'
            ]
            
            model_dir = None
            available_dirs = []
            
            # Primero, verificar qué directorios existen
            for directory in possible_dirs:
                if os.path.exists(directory):
                    try:
                        contents = os.listdir(directory)
                        available_dirs.append((directory, len(contents)))
                        if contents:  # Si el directorio no está vacío
                            model_dir = directory
                            logger.info(f"Directorio de modelos encontrado: {directory} con {len(contents)} archivos")
                            break
                    except PermissionError:
                        logger.warning(f"Sin permisos para acceder a {directory}")
                        continue
            
            if not model_dir:
                logger.warning("No se encontraron directorios de modelos válidos")
                logger.info(f"Directorios verificados: {[d[0] for d in available_dirs]}")
                logger.info("Creando modelos de respaldo...")
                self.create_fallback_models()
                return
            
            # Cargar modelos disponibles
            model_files = {
                'linear': 'modelo_linear.pkl',
                'ridge': 'modelo_ridge.pkl', 
                'lasso': 'modelo_lasso.pkl',
                'random_forest': 'modelo_random_forest.pkl',
                'gradient_boost': 'modelo_gradient_boost.pkl',
                'neural_net': 'modelo_neural_net.pkl'
            }
            
            loaded_models = 0
            for name, filename in model_files.items():
                model_path = os.path.join(model_dir, filename)
                if os.path.exists(model_path):
                    try:
                        self.models[name] = joblib.load(model_path)
                        logger.info(f"✅ Modelo {name} cargado desde {model_path}")
                        loaded_models += 1
                    except Exception as e:
                        logger.error(f"❌ Error cargando modelo {name}: {e}")
                else:
                    logger.warning(f"⚠️ Modelo no encontrado: {model_path}")
            
            # Cargar escaladores
            for name in self.models.keys():
                scaler_path = os.path.join(model_dir, f'scaler_{name}.pkl')
                if os.path.exists(scaler_path):
                    try:
                        self.scalers[name] = joblib.load(scaler_path)
                        logger.info(f"✅ Escalador {name} cargado")
                    except Exception as e:
                        logger.warning(f"⚠️ Error cargando escalador {name}: {e}")
                        
                poly_path = os.path.join(model_dir, f'poly_{name}.pkl')
                if os.path.exists(poly_path):
                    try:
                        self.poly_features[name] = joblib.load(poly_path)
                        logger.info(f"✅ Características polinomiales {name} cargadas")
                    except Exception as e:
                        logger.warning(f"⚠️ Error cargando poly {name}: {e}")
            
            # Cargar scores de modelos
            scores_path = os.path.join(model_dir, 'model_scores.pkl')
            if os.path.exists(scores_path):
                try:
                    self.model_scores = joblib.load(scores_path)
                    logger.info(f"✅ Scores de modelos cargados")
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando scores: {e}")
                    # Scores por defecto
                    for name in self.models.keys():
                        self.model_scores[name] = {
                            'prediction_confidence': 85.0,
                            'r2_score': 0.85,
                            'mae': 1.2,
                            'rmse': 1.8
                        }
            else:
                # Scores por defecto
                for name in self.models.keys():
                    self.model_scores[name] = {
                        'prediction_confidence': 85.0,
                        'r2_score': 0.85,
                        'mae': 1.2,
                        'rmse': 1.8
                    }
            
            # Cargar columnas de características
            features_path = os.path.join(model_dir, 'feature_columns.pkl')
            if os.path.exists(features_path):
                try:
                    self.feature_columns = joblib.load(features_path)
                    logger.info(f"✅ Columnas de características cargadas: {len(self.feature_columns)}")
                except Exception as e:
                    logger.warning(f"⚠️ Error cargando columnas: {e}")
                    self.feature_columns = [
                        'year_since_1990', 'decade', 'month', 'season',
                        'latitude', 'longitude', 'altitude_norm',
                        'coastal', 'highland', 'jungle',
                        'el_nino_index', 'solar_cycle',
                        'linear_trend', 'quadratic_trend'
                    ]
            else:
                self.feature_columns = [
                    'year_since_1990', 'decade', 'month', 'season',
                    'latitude', 'longitude', 'altitude_norm',
                    'coastal', 'highland', 'jungle',
                    'el_nino_index', 'solar_cycle',
                    'linear_trend', 'quadratic_trend'
                ]
            
            logger.info(f"📊 Resumen de carga: {loaded_models}/{len(model_files)} modelos cargados")
            
            # Si no se cargó ningún modelo, crear modelos de respaldo
            if loaded_models == 0:
                logger.warning("❌ No se pudo cargar ningún modelo, creando modelos de respaldo")
                self.create_fallback_models()
            
        except Exception as e:
            logger.error(f"❌ Error crítico cargando modelos: {e}")
            logger.info("Creando modelos de respaldo como última opción")
            self.create_fallback_models()
    def create_fallback_models(self):
        """Crea modelos básicos de respaldo"""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        
        # Crear datos sintéticos para entrenamiento de respaldo
        np.random.seed(42)
        years = np.array(range(1990, 2024))
        n_samples = len(years) * 12 * 5  # 34 años * 12 meses * 5 regiones aprox
        
        X = np.random.randn(n_samples, 14)  # 14 características
        # Generar temperaturas realistas para Perú (12-30°C)
        base_temp = 18.5
        y = base_temp + X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.normal(0, 1, n_samples)
        y = np.clip(y, 12, 30)  # Rango válido para Perú
        
        # Entrenar modelos básicos
        models_to_create = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        for name, model in models_to_create.items():
            try:
                model.fit(X, y)
                self.models[name] = model
                self.model_scores[name] = {
                    'prediction_confidence': 75.0,
                    'r2_score': 0.75,
                    'mae': 1.5,
                    'rmse': 2.0
                }
                logger.info(f"Modelo de respaldo {name} creado")
            except Exception as e:
                logger.error(f"Error creando modelo {name}: {e}")
        
        self.feature_columns = [f'feature_{i}' for i in range(14)]
        logger.info("Modelos de respaldo creados")
        
    def predict_temperature(self, features, model_name='neural_net', region='peru'):
            
            try:
                # Log inicial
                logger.info(f"🔍 Iniciando predicción: modelo={model_name}, region={region}")
                
                if model_name not in self.models:
                    logger.error(f"❌ Modelo {model_name} no existe")
                    available = list(self.models.keys())
                    logger.info(f"Modelos disponibles: {available}")
                    if available:
                        model_name = available[0]
                        logger.info(f"→ Usando modelo alternativo: {model_name}")
                    else:
                        logger.error("❌ NO HAY MODELOS CARGADOS")
                        return self.fallback_prediction(features, region)
                
                # Preparar características
                X = self.prepare_features(features)
                logger.info(f"📊 Features shape: {X.shape}, valores: {X[0][:5]}...")
                
                # Verificar transformaciones
                original_shape = X.shape
                if model_name in self.poly_features:
                    X = self.poly_features[model_name].transform(X)
                    logger.info(f"🔄 Poly transform: {original_shape} → {X.shape}")
                
                if model_name in self.scalers:
                    X = self.scalers[model_name].transform(X)
                    logger.info(f"📏 Scaled: min={X.min():.2f}, max={X.max():.2f}")
                
                # PREDICCIÓN REAL
                prediction = self.models[model_name].predict(X)[0]
                logger.info(f"✅ Predicción RAW del modelo: {prediction:.2f}°C")
                
                # Ajustes
                adjusted = self.apply_regional_adjustment(prediction, region, features)
                logger.info(f"🔧 Después de ajuste regional: {adjusted:.2f}°C")
                
                confidence = self.calculate_prediction_confidence(model_name, features, adjusted)
                final = self.validate_prediction_range(adjusted, region)
                
                logger.info(f"🎯 Predicción FINAL: {final:.2f}°C (confianza: {confidence:.1f}%)")
                
                return float(final), float(confidence)
                
            except Exception as e:
                logger.error(f"❌ ERROR CRÍTICO en predicción: {e}")
                logger.exception(e)  # Stack trace completo
                return self.fallback_prediction(features, region)
        
            predictions = []
            
            for year in range(start_year, end_year + 1):
                features = {
                    'year': year,
                    'year_since_1990': year - 1990,
                    'decade': (year - 1990) // 10,
                    'month': month,
                    'season': ((month - 1) // 3) + 1,
                    'latitude': regionCoords.get(region, {}).get('lat', -12),
                    'longitude': regionCoords.get(region, {}).get('lng', -75),
                    'altitude': self.get_region_altitude(region),
                    'altitude_norm': (self.get_region_altitude(region) - 1500) / 1500,
                    'coastal': 1 if region in ['lima', 'trujillo'] else 0,
                    'highland': 1 if region in ['cusco', 'arequipa'] else 0,
                    'jungle': 1 if region == 'iquitos' else 0,
                    'el_nino_index': np.sin(2 * np.pi * (year - 1990) / 4.5),
                    'solar_cycle': np.sin(2 * np.pi * (year - 1990) / 11),
                    'linear_trend': year - 1990,
                    'quadratic_trend': (year - 1990) ** 2,
                    'region': region
                }
                
                temp, confidence = self.predict_temperature(features, model_name, region)
                predictions.append({
                    'year': year,
                    'temperature': temp,
                    'confidence': confidence
                })
            
            return predictions

    def get_region_altitude(self, region):
        altitudes = {
            'lima': 154, 'cusco': 3399, 'arequipa': 2335,
            'iquitos': 106, 'trujillo': 34, 'peru': 500
        }
        return altitudes.get(region, 500)

    def load_historical_data(self):
        """Carga datos históricos para comparación y análisis"""
        try:
            # Buscar archivo de datos históricos
            data_files = [
                'data/peru_climate_data.csv',
                '../data/peru_climate_data.csv',
                'peru_climate_data.csv',
                'historical_data.csv'
            ]
            
            data_file = None
            for file_path in data_files:
                if os.path.exists(file_path):
                    data_file = file_path
                    break
            
            if data_file:
                df = pd.read_csv(data_file)
                logger.info(f"Datos históricos cargados: {len(df)} registros")
                
                # Procesar datos por región
                self.process_historical_by_region(df)
            else:
                logger.warning("No se encontró archivo de datos históricos")
                self.generate_synthetic_historical_data()
                
        except Exception as e:
            logger.error(f"Error cargando datos históricos: {e}")
            self.generate_synthetic_historical_data()

    def process_historical_by_region(self, df):
        """Procesa datos históricos agrupados por región"""
        try:
            # Mapeo de ciudades a regiones
            region_mapping = {
                'lima': ['lima', 'callao', 'ancón'],
                'cusco': ['cusco', 'machu picchu', 'pisac'],
                'arequipa': ['arequipa', 'chivay', 'camaná'],
                'iquitos': ['iquitos', 'pucallpa', 'tarapoto'],
                'trujillo': ['trujillo', 'chiclayo', 'cajamarca']
            }
            
            # Asegurar que tenemos columnas necesarias
            if 'year' not in df.columns and 'FECHA' in df.columns:
                df['year'] = pd.to_datetime(df['FECHA'], errors='coerce').dt.year
            
            if 'temperature' not in df.columns and 'TEMP' in df.columns:
                df['temperature'] = pd.to_numeric(df['TEMP'], errors='coerce')
            
            # Procesar cada región
            for region, cities in region_mapping.items():
                region_data = []
                
                if 'city' in df.columns:
                    city_mask = df['city'].str.lower().str.contains('|'.join(cities), na=False)
                    region_df = df[city_mask]
                elif 'DEPARTAMENTO' in df.columns:
                    dept_name = region.upper() if region != 'lima' else 'LIMA'
                    region_df = df[df['DEPARTAMENTO'].str.contains(dept_name, na=False)]
                else:
                    # Si no hay información de ubicación, usar todos los datos
                    region_df = df.sample(frac=0.2, random_state=42)
                
                if len(region_df) > 0:
                    # Agrupar por año y calcular promedio
                    yearly_data = region_df.groupby('year')['temperature'].agg(['mean', 'std', 'count']).reset_index()
                    yearly_data = yearly_data[yearly_data['count'] >= 3]  # Al menos 3 registros por año
                    
                    self.historical_data[region] = {
                        'years': yearly_data['year'].tolist(),
                        'temperatures': yearly_data['mean'].round(1).tolist(),
                        'temperature_std': yearly_data['std'].fillna(0).round(1).tolist(),
                        'data_count': yearly_data['count'].tolist()
                    }
                    
                    logger.info(f"Datos procesados para {region}: {len(yearly_data)} años")
            
            # Agregar promedio nacional
            if 'peru' not in self.historical_data:
                national_data = df.groupby('year')['temperature'].agg(['mean', 'std', 'count']).reset_index()
                national_data = national_data[national_data['count'] >= 10]
                
                self.historical_data['peru'] = {
                    'years': national_data['year'].tolist(),
                    'temperatures': national_data['mean'].round(1).tolist(),
                    'temperature_std': national_data['std'].fillna(0).round(1).tolist(),
                    'data_count': national_data['count'].tolist()
                }
                
                logger.info(f"Datos nacionales procesados: {len(national_data)} años")
                
        except Exception as e:
            logger.error(f"Error procesando datos históricos: {e}")
            self.generate_synthetic_historical_data()

    def generate_synthetic_historical_data(self):
        """Genera datos históricos sintéticos realistas"""
        logger.info("Generando datos históricos sintéticos")
        
        regions = {
            'peru': {'base_temp': 18.5, 'variation': 1.2, 'warming_rate': 0.03},
            'lima': {'base_temp': 19.0, 'variation': 1.0, 'warming_rate': 0.025},
            'cusco': {'base_temp': 15.0, 'variation': 1.8, 'warming_rate': 0.035},
            'arequipa': {'base_temp': 18.0, 'variation': 1.3, 'warming_rate': 0.028},
            'iquitos': {'base_temp': 26.0, 'variation': 0.8, 'warming_rate': 0.032},
            'trujillo': {'base_temp': 22.0, 'variation': 1.1, 'warming_rate': 0.030}
        }
        
        np.random.seed(42)
        
        for region, config in regions.items():
            years = list(range(1990, 2024))
            temperatures = []
            temperature_std = []
            
            for i, year in enumerate(years):
                # Tendencia de calentamiento
                warming_trend = i * config['warming_rate']
                
                # Variaciones cíclicas (El Niño, PDO, etc.)
                enso_cycle = np.sin(2 * np.pi * i / 4.5) * 0.6  # Ciclo ENSO ~4.5 años
                decadal_cycle = np.sin(2 * np.pi * i / 15) * 0.4  # Variación decenal
                
                # Ruido aleatorio
                noise = np.random.normal(0, config['variation'])
                
                # Temperatura final
                temp = config['base_temp'] + warming_trend + enso_cycle + decadal_cycle + noise
                temp = max(10, min(35, temp))  # Rango físico válido
                
                temperatures.append(round(temp, 1))
                temperature_std.append(round(abs(noise), 1))
            
            self.historical_data[region] = {
                'years': years,
                'temperatures': temperatures,
                'temperature_std': temperature_std,
                'data_count': [12] * len(years)  # 12 meses por año
            }
        
        logger.info(f"Datos sintéticos generados para {len(regions)} regiones")

    def predict_temperature(self, features, model_name='neural_net', region='peru'):
        try:
            # Verificar modelo disponible
            if model_name not in self.models:
                available_models = list(self.models.keys())
                if available_models:
                    logger.warning(f"Modelo {model_name} no disponible, usando {available_models[0]}")
                    model_name = available_models[0]
                else:
                    logger.error("No hay modelos disponibles")
                    return self.fallback_prediction(features, region)
            
            # Preparar características
            X = self.prepare_features(features)
            
            # Validar que X no tenga valores inválidos
            if np.isnan(X).any() or np.isinf(X).any():
                logger.error("Features contienen valores inválidos")
                return self.fallback_prediction(features, region)
            
            # Aplicar transformaciones
            if model_name in self.poly_features:
                X = self.poly_features[model_name].transform(X)
            
            if model_name in self.scalers:
                X = self.scalers[model_name].transform(X)
            
            # Hacer predicción
            prediction = self.models[model_name].predict(X)[0]
            logger.info(f"Prediccion raw del modelo: {prediction:.2f}C")
            
            # Aplicar ajustes regionales
            adjusted_prediction = self.apply_regional_adjustment(prediction, region, features)
            
            # Obtener confianza
            confidence = self.calculate_prediction_confidence(
                model_name, features, adjusted_prediction
            )
            
            # Validar y ajustar rango
            final_prediction = self.validate_prediction_range(adjusted_prediction, region)
            
            return float(final_prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Error en prediccion: {e}")
            logger.exception(e)
            return self.fallback_prediction(features, region)
        
    def prepare_features(self, features):
        
        try:
            # Validar que features es un dict
            if not isinstance(features, dict):
                logger.error(f"Features debe ser dict, recibido: {type(features)}")
                return self.get_default_features()
            
            # Validar que tenemos las columnas esperadas
            if not self.feature_columns:
                logger.error("feature_columns no está inicializado")
                return self.get_default_features()
            
            # Si el modelo es simple (pocas columnas)
            if len(self.feature_columns) <= 5:
                X = np.array([[
                    float(features.get('year_since_1990', 30)),
                    float(features.get('month', 6)),
                    float(features.get('latitude', -12))
                ]])
                return X
            
            # Modelo completo - construir vector ordenado
            X = np.zeros((1, len(self.feature_columns)))
            
            for i, feature_name in enumerate(self.feature_columns):
                if feature_name in features:
                    try:
                        value = float(features[feature_name])
                        # Validar que el valor sea razonable
                        if np.isnan(value) or np.isinf(value):
                            X[0, i] = self.get_default_value(feature_name)
                        else:
                            X[0, i] = value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error convirtiendo {feature_name}: {e}")
                        X[0, i] = self.get_default_value(feature_name)
                else:
                    X[0, i] = self.get_default_value(feature_name)
            
            # Validación final
            if np.isnan(X).any() or np.isinf(X).any():
                logger.warning("Features contienen NaN o Inf despues de construccion, usando defaults")
                return self.get_default_features()
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            logger.exception(e)
            return self.get_default_features()
    
    def get_default_value(self, feature_name):
      
        defaults = {
            'year_since_1990': 34, 
            'decade': 3, 
            'month': 6, 
            'season': 2,
            'latitude': -12.0, 
            'longitude': -75.0, 
            'altitude': 500,
            'altitude_norm': 0.0,
            'coastal': 0, 
            'highland': 0, 
            'jungle': 0,
            'el_nino_index': 0.0, 
            'solar_cycle': 0.0,
            'linear_trend': 34, 
            'quadratic_trend': 1156,
            'year': 2024
        }
        return defaults.get(feature_name, 0.0)

    def get_default_features(self):
        """Retorna matriz de features por defecto"""
        if len(self.feature_columns) <= 5:
            return np.array([[34, 6, -12]])
        return np.array([[self.get_default_value(col) for col in self.feature_columns]])


    def apply_regional_adjustment(self, prediction, region, features):
        
        regional_adjustments = {
            'lima': -0.5,     
            'cusco': -3.0,    
            'arequipa': -0.2, 
            'iquitos': +6.0,  
            'trujillo': +2.0, 
            'peru': 0.0       
        }
        
        base_adjustment = regional_adjustments.get(region, 0.0)
        
        # Ajuste adicional por altitud si está disponible
        altitude = features.get('altitude', 0)
        if altitude > 2000:
            altitude_adjustment = -(altitude - 2000) / 1000 * 1.5  # -1.5°C por cada 1000m
            base_adjustment += altitude_adjustment
        
        # Ajuste estacional
        month = features.get('month', 6)
        if region in ['lima', 'trujillo']:  # Costa
            seasonal_adj = -0.5 if 6 <= month <= 9 else 0.5  # Invierno más fresco
        elif region in ['cusco', 'arequipa']:  # Sierra
            seasonal_adj = -1.0 if 6 <= month <= 8 else 0.5   # Invierno más frío
        else:  # Selva y nacional
            seasonal_adj = 0.2 if 12 <= month <= 3 else -0.2  # Verano más cálido
        
        return prediction + base_adjustment + seasonal_adj

    def calculate_prediction_confidence(self, model_name, features, prediction):
      
        base_confidence = self.model_scores.get(model_name, {}).get('prediction_confidence', 80.0)
        
        # Ajustar confianza según factores
        year = features.get('year', 2030)
        
        # Reducir confianza para años muy futuros
        if year > 2050:
            time_penalty = min((year - 2050) * 2, 20)  # Hasta 20% de penalización
            base_confidence -= time_penalty
        
        # Reducir confianza para temperaturas extremas
        if prediction < 10 or prediction > 35:
            base_confidence *= 0.7
        
        # Aumentar confianza si está en rango histórico normal para la región
        expected_ranges = {
            'lima': (16, 22), 'cusco': (10, 18), 'arequipa': (15, 21),
            'iquitos': (24, 28), 'trujillo': (18, 26), 'peru': (15, 25)
        }
        
        region = features.get('region', 'peru')
        if region in expected_ranges:
            min_temp, max_temp = expected_ranges[region]
            if min_temp <= prediction <= max_temp:
                base_confidence = min(base_confidence + 5, 95)
        
        return max(60, min(95, base_confidence))

    def validate_prediction_range(self, prediction, region):
        
        # Rangos absolutos por región (°C)
        absolute_ranges = {
            'lima': (12, 28), 'cusco': (5, 25), 'arequipa': (8, 28),
            'iquitos': (20, 35), 'trujillo': (15, 32), 'peru': (8, 35)
        }
        
        min_temp, max_temp = absolute_ranges.get(region, (8, 35))
        
        if prediction < min_temp:
            logger.warning(f"Predicción {prediction}°C muy baja para {region}, ajustando a {min_temp}°C")
            return min_temp
        elif prediction > max_temp:
            logger.warning(f"Predicción {prediction}°C muy alta para {region}, ajustando a {max_temp}°C")
            return max_temp
        
        return prediction

    def fallback_prediction(self, features, region='peru'):
        """Predicción de respaldo usando modelos estadísticos simples"""
        try:
            year = features.get('year', 2024)
            month = features.get('month', 6)
            years_since_1990 = year - 1990
            
            # Temperaturas base históricas por región
            base_temps = {
                'peru': 18.5, 'lima': 19.0, 'cusco': 15.0,
                'arequipa': 18.0, 'iquitos': 26.0, 'trujillo': 22.0
            }
            
            base_temp = base_temps.get(region, 18.5)
            
            # Tendencia de calentamiento global
            warming_trend = years_since_1990 * 0.03
            
            # Variación estacional
            seasonal_variation = np.cos((month - 1) * np.pi / 6) * 1.5
            
            # Variación por El Niño (ciclo ~4.5 años)
            enso_effect = np.sin(2 * np.pi * years_since_1990 / 4.5) * 0.8
            
            prediction = base_temp + warming_trend + seasonal_variation + enso_effect
            
            # Validar rango
            prediction = self.validate_prediction_range(prediction, region)
            
            return prediction, 75.0  # Confianza moderada para fallback
            
        except Exception as e:
            logger.error(f"Error en fallback: {e}")
            return 18.5, 70.0

    def get_historical_data(self, region='peru'):
        """Obtiene datos históricos para una región específica"""
        region = region.lower()
        if region in self.historical_data:
            return self.historical_data[region]
        else:
            logger.warning(f"No hay datos históricos para {region}")
            return {
                'years': list(range(1990, 2024)),
                'temperatures': [18.5] * 34,
                'temperature_std': [1.0] * 34,
                'data_count': [12] * 34
            }

    def get_prediction_map_data(self, year=2030, model_name='neural_net', month=6):
        """Genera datos completos para mapa de predicciones"""
        cities_data = {
            'Lima': {
                'lat': -12.0464, 'lng': -77.0428, 'region': 'lima',
                'altitude': 154, 'population': 10750000, 'coastal': 1, 'highland': 0, 'jungle': 0
            },
            'Cusco': {
                'lat': -13.5319, 'lng': -71.9675, 'region': 'cusco',
                'altitude': 3399, 'population': 428450, 'coastal': 0, 'highland': 1, 'jungle': 0
            },
            'Arequipa': {
                'lat': -16.4090, 'lng': -71.5375, 'region': 'arequipa',
                'altitude': 2335, 'population': 1080000, 'coastal': 0, 'highland': 1, 'jungle': 0
            },
            'Iquitos': {
                'lat': -3.7437, 'lng': -73.2516, 'region': 'iquitos',
                'altitude': 106, 'population': 437620, 'coastal': 0, 'highland': 0, 'jungle': 1
            },
            'Trujillo': {
                'lat': -8.1116, 'lng': -79.0290, 'region': 'trujillo',
                'altitude': 34, 'population': 919899, 'coastal': 1, 'highland': 0, 'jungle': 0
            },
            'Chiclayo': {
                'lat': -6.7714, 'lng': -79.8374, 'region': 'trujillo',
                'altitude': 57, 'population': 600440, 'coastal': 1, 'highland': 0, 'jungle': 0
            },
            'Huancayo': {
                'lat': -12.0653, 'lng': -75.2049, 'region': 'cusco',
                'altitude': 3259, 'population': 545615, 'coastal': 0, 'highland': 1, 'jungle': 0
            },
            'Piura': {
                'lat': -5.1945, 'lng': -80.6328, 'region': 'trujillo',
                'altitude': 30, 'population': 484475, 'coastal': 1, 'highland': 0, 'jungle': 0
            }
        }
        
        map_data = []
        
        for city, data in cities_data.items():
            # Crear vector de características
            features = {
                'year': year,
                'year_since_1990': year - 1990,
                'month': month,
                'season': ((month - 1) // 3) + 1,
                'latitude': data['lat'],
                'longitude': data['lng'],
                'altitude': data['altitude'],
                'altitude_norm': (data['altitude'] - 1500) / 1500,  # Normalización aproximada
                'coastal': data['coastal'],
                'highland': data['highland'],
                'jungle': data['jungle'],
                'el_nino_index': np.sin(2 * np.pi * (year - 1990) / 4.5),
                'solar_cycle': np.sin(2 * np.pi * (year - 1990) / 11),
                'linear_trend': year - 1990,
                'quadratic_trend': (year - 1990) ** 2,
                'decade': (year - 1990) // 10,
                'region': data['region']
            }
            
            # Realizar predicción
            temp, confidence = self.predict_temperature(features, model_name, data['region'])
            
            # Determinar nivel de alerta
            alert_level = self.get_temperature_alert_level(temp, data['region'])
            
            map_data.append({
                'city': city,
                'latitude': data['lat'],
                'longitude': data['lng'],
                'temperature': round(temp, 1),
                'confidence': round(confidence, 1),
                'region': data['region'],
                'altitude': data['altitude'],
                'population': data['population'],
                'alert_level': alert_level,
                'climate_zone': self.get_climate_zone(data)
            })
        
        return map_data

    def get_temperature_alert_level(self, temperature, region):
        alert_thresholds = {
            'lima': {'extreme_high': 28, 'high': 25, 'low': 15, 'extreme_low': 12},
            'cusco': {'extreme_high': 22, 'high': 20, 'low': 8, 'extreme_low': 5},
            'arequipa': {'extreme_high': 25, 'high': 22, 'low': 10, 'extreme_low': 7},
            'iquitos': {'extreme_high': 32, 'high': 30, 'low': 22, 'extreme_low': 20},
            'trujillo': {'extreme_high': 30, 'high': 28, 'low': 16, 'extreme_low': 14}
        }
        
        thresholds = alert_thresholds.get(region, alert_thresholds['lima'])
        
        if temperature >= thresholds['extreme_high']:
            return 'extreme_high'
        elif temperature >= thresholds['high']:
            return 'high'
        elif temperature <= thresholds['extreme_low']:
            return 'extreme_low'
        elif temperature <= thresholds['low']:
            return 'low'
        else:
            return 'normal'

    def get_climate_zone(self, city_data):
        """Determina zona climática basada en características geográficas"""
        if city_data['coastal']:
            return 'Costa'
        elif city_data['highland']:
            return 'Sierra'
        elif city_data['jungle']:
            return 'Selva'
        else:
            return 'Mixta'

    def generate_predictions_csv(self, year, model_name, regions, months=None):
        """Genera CSV con predicciones detalladas"""
        if months is None:
            months = list(range(1, 13))  # Todos los meses
        
        predictions = []
        
        for region in regions:
            region_coords = {
                'lima': {'lat': -12.0464, 'lng': -77.0428, 'alt': 154},
                'cusco': {'lat': -13.5319, 'lng': -71.9675, 'alt': 3399},
                'arequipa': {'lat': -16.4090, 'lng': -71.5375, 'alt': 2335},
                'iquitos': {'lat': -3.7437, 'lng': -73.2516, 'alt': 106},
                'trujillo': {'lat': -8.1116, 'lng': -79.0290, 'alt': 34},
                'peru': {'lat': -9.1900, 'lng': -75.0152, 'alt': 500}
            }
            
            coords = region_coords.get(region, region_coords['peru'])
            
            for month in months:
                features = {
                    'year': year,
                    'year_since_1990': year - 1990,
                    'month': month,
                    'season': ((month - 1) // 3) + 1,
                    'latitude': coords['lat'],
                    'longitude': coords['lng'],
                    'altitude': coords['alt'],
                    'altitude_norm': (coords['alt'] - 1500) / 1500,
                    'coastal': 1 if region in ['lima', 'trujillo'] else 0,
                    'highland': 1 if region in ['cusco', 'arequipa'] else 0,
                    'jungle': 1 if region == 'iquitos' else 0,
                    'el_nino_index': np.sin(2 * np.pi * (year - 1990) / 4.5),
                    'solar_cycle': np.sin(2 * np.pi * (year - 1990) / 11),
                    'linear_trend': year - 1990,
                    'quadratic_trend': (year - 1990) ** 2,
                    'decade': (year - 1990) // 10,
                    'region': region
                }
                
                temp, confidence = self.predict_temperature(features, model_name, region)
                alert_level = self.get_temperature_alert_level(temp, region)
                
                # Nombres de meses en español
                month_names = [
                    'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                    'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
                ]
                
                predictions.append({
                    'region': region.upper(),
                    'region_name': region.replace('_', ' ').title(),
                    'year': year,
                    'month': month,
                    'month_name': month_names[month - 1],
                    'season': features['season'],
                    'predicted_temperature': round(temp, 1),
                    'confidence_percent': round(confidence, 1),
                    'alert_level': alert_level,
                    'model_used': model_name,
                    'latitude': coords['lat'],
                    'longitude': coords['lng'],
                    'altitude_m': coords['alt'],
                    'climate_zone': 'Costa' if region in ['lima', 'trujillo'] else 'Sierra' if region in ['cusco', 'arequipa'] else 'Selva' if region == 'iquitos' else 'Nacional',
                    'generation_timestamp': datetime.now().isoformat()
                })
        
        return predictions


# Inicializar predictor
climate_predictor = AdvancedClimatePredictor()


# ===============================
# RUTAS DE LA API
# ===============================

from flask import render_template

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error cargando HTML: {e}", 500


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint mejorado para predicción de temperatura"""
    try:
        data = request.get_json(force=True)
        
        # Extraer y validar parámetros
        model_name = data.get('model', 'neural_net')
        region = data.get('region', 'peru')
        features = data.get('features', {})
        
        if not features:
            return jsonify({
                'error': 'Se requieren características para la predicción',
                'required_features': climate_predictor.feature_columns[:5],
                'status': 'error'
            }), 400
        
        # Validar modelo disponible
        if model_name not in climate_predictor.models:
            available_models = list(climate_predictor.models.keys())
            return jsonify({
                'error': f'Modelo {model_name} no disponible',
                'available_models': available_models,
                'status': 'error'
            }), 400
        
        # Realizar predicción
        temperature, confidence = climate_predictor.predict_temperature(
            features, model_name, region
        )
        
        # Determinar nivel de alerta
        alert_level = climate_predictor.get_temperature_alert_level(temperature, region)
        
        # Preparar respuesta completa
        response_data = {
            'predicted_temperature': round(temperature, 1),
            'confidence': round(confidence, 1),
            'model_used': model_name,
            'region': region,
            'alert_level': alert_level,
            'climate_context': get_climate_context(temperature, region),
            'prediction_date': features.get('year', 2030),
            'prediction_month': features.get('month', 6),
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'model_info': climate_predictor.model_scores.get(model_name, {})
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Información detallada de modelos disponibles"""
    try:
        return jsonify({
            'available_models': list(climate_predictor.models.keys()),
            'model_scores': climate_predictor.model_scores,
            'total_models': len(climate_predictor.models),
            'feature_count': len(climate_predictor.feature_columns),
            'feature_names': climate_predictor.feature_columns,
            'best_model': get_best_model(),
            'status': 'success',
            'system_info': {
                'version': '2.0',
                'regions_supported': list(climate_predictor.historical_data.keys()),
                'prediction_range': {'min_year': 2024, 'max_year': 2100}
            }
        })
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Estado de salud detallado del sistema"""
    try:
        system_health = {
            'status': 'healthy',
            'models_loaded': len(climate_predictor.models),
            'scalers_loaded': len(climate_predictor.scalers),
            'historical_regions': len(climate_predictor.historical_data),
            'api_version': '2.0 - Advanced',
            'ready': len(climate_predictor.models) > 0,
            'capabilities': {
                'historical_analysis': len(climate_predictor.historical_data) > 0,
                'map_predictions': True,
                'csv_export': True,
                'multi_model_support': len(climate_predictor.models) > 1,
                'regional_analysis': True
            },
            'performance': {
                'avg_confidence': calculate_avg_confidence(),
                'supported_regions': list(climate_predictor.historical_data.keys()),
                'data_coverage': '1990-2023'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Determinar estado general
        if len(climate_predictor.models) == 0:
            system_health['status'] = 'degraded'
            system_health['warning'] = 'No hay modelos cargados'
        elif len(climate_predictor.historical_data) == 0:
            system_health['status'] = 'limited'
            system_health['warning'] = 'Datos históricos limitados'
        
        return jsonify(system_health)
        
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/debug/features', methods=['POST'])
def debug_features():
    """Endpoint de debug para verificar features"""
    try:
        data = request.get_json(force=True)
        features = data.get('features', {})
        
        # Preparar features
        X = climate_predictor.prepare_features(features)
        
        return jsonify({
            'received_features': features,
            'expected_columns': climate_predictor.feature_columns,
            'prepared_array_shape': X.shape,
            'prepared_array': X.tolist(),
            'available_models': list(climate_predictor.models.keys()),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
        
@app.route('/historical/<region>', methods=['GET'])
def get_historical(region):
    """Datos históricos detallados por región"""
    try:
        region = region.lower().strip()
        historical_data = climate_predictor.get_historical_data(region)
        
        if not historical_data or not historical_data.get('years'):
            return jsonify({
                'error': f'No hay datos históricos para la región: {region}',
                'available_regions': list(climate_predictor.historical_data.keys()),
                'status': 'error'
            }), 404
        
        # Calcular estadísticas adicionales
        temps = historical_data['temperatures']
        statistics = {
            'mean_temperature': round(np.mean(temps), 1),
            'min_temperature': round(np.min(temps), 1),
            'max_temperature': round(np.max(temps), 1),
            'temperature_trend': calculate_temperature_trend(
                historical_data['years'], temps
            ),
            'std_deviation': round(np.std(temps), 1),
            'warming_rate_per_decade': calculate_warming_rate(
                historical_data['years'], temps
            )
        }
        
        return jsonify({
            'region': region,
            'region_display_name': region.replace('_', ' ').title(),
            'years': historical_data['years'],
            'temperatures': historical_data['temperatures'],
            'temperature_std': historical_data.get('temperature_std', []),
            'data_count': historical_data.get('data_count', []),
            'data_points': len(historical_data['years']),
            'date_range': {
                'start': min(historical_data['years']),
                'end': max(historical_data['years'])
            },
            'statistics': statistics,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo datos históricos: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/predict_map', methods=['GET'])
def predict_map():
    """Datos completos para mapa interactivo de predicciones"""
    try:
        # Parámetros de consulta
        year = int(request.args.get('year', 2030))
        model_name = request.args.get('model', 'neural_net')
        month = int(request.args.get('month', 6))
        
        # Validar parámetros
        if year < 2024 or year > 2100:
            return jsonify({
                'error': f'Año {year} fuera de rango válido (2024-2100)',
                'status': 'error'
            }), 400
        
        if model_name not in climate_predictor.models:
            return jsonify({
                'error': f'Modelo {model_name} no disponible',
                'available_models': list(climate_predictor.models.keys()),
                'status': 'error'
            }), 400
        
        # Generar datos del mapa
        map_data = climate_predictor.get_prediction_map_data(year, model_name, month)
        
        # Calcular estadísticas del mapa
        temperatures = [city['temperature'] for city in map_data]
        confidences = [city['confidence'] for city in map_data]
        
        map_statistics = {
            'avg_temperature': round(np.mean(temperatures), 1),
            'max_temperature': round(np.max(temperatures), 1),
            'min_temperature': round(np.min(temperatures), 1),
            'avg_confidence': round(np.mean(confidences), 1),
            'temperature_range': round(np.max(temperatures) - np.min(temperatures), 1),
            'cities_high_alert': len([c for c in map_data if c['alert_level'] in ['high', 'extreme_high']]),
            'cities_low_alert': len([c for c in map_data if c['alert_level'] in ['low', 'extreme_low']])
        }
        
        return jsonify({
            'year': year,
            'month': month,
            'month_name': [
                'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
            ][month - 1],
            'model_used': model_name,
            'predictions': map_data,
            'total_locations': len(map_data),
            'statistics': map_statistics,
            'legend': {
                'extreme_high': 'Calor Extremo',
                'high': 'Temperatura Alta', 
                'normal': 'Normal',
                'low': 'Temperatura Baja',
                'extreme_low': 'Frío Extremo'
            },
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error generando mapa: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

ALLOWED_MODELS = {'linear', 'ridge', 'lasso', 'random_forest', 'gradient_boost', 'neural_net'}

@app.route('/predict_range', methods=['POST'])
def predict_range():
    """Nuevo endpoint para predicciones de rango"""
    try:
        data = request.get_json(force=True)
        
        model_name = data.get('model', 'neural_net')
        region = data.get('region', 'peru')
        end_year = data.get('year', 2030)
        start_year = data.get('start_year', 2024)
        month = data.get('month', 6)
        
        predictions = climate_predictor.predict_temperature_range(
            start_year, end_year, month, region, model_name
        )
        
        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'model_used': model_name,
            'region': region,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en predicción de rango: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500# AGREGAR ESTAS NUEVAS RUTAS Y FUNCIONES A TU server.py

@app.route('/health_recommendations', methods=['POST'])
def get_health_recommendations():
    """Recomendaciones de salud basadas en temperatura y calidad del aire"""
    try:
        data = request.get_json(force=True)
        temperature = data.get('temperature', 20)
        region = data.get('region', 'peru')
        month = data.get('month', 6)
        alert_level = data.get('alert_level', 'normal')
        
        # Calcular índices de calidad del aire estimados
        aqi = estimate_aqi(temperature, region, month)
        uv_index = estimate_uv_index(region, month)
        
        recommendations = {
            'temperature_alerts': get_temperature_health_alerts(temperature, alert_level),
            'air_quality': {
                'aqi': aqi,
                'level': get_aqi_level(aqi),
                'advice': get_aqi_advice(aqi)
            },
            'uv_protection': {
                'index': uv_index,
                'level': get_uv_level(uv_index),
                'advice': get_uv_advice(uv_index)
            },
            'general_health': get_general_health_tips(temperature, region, month),
            'vulnerable_groups': get_vulnerable_groups_advice(temperature, alert_level),
            'emergency_numbers': {
                'peru': '116',  # SAMU
                'fire': '116',
                'police': '105',
                'defensa_civil': '115'
            }
        }
        
        return jsonify({
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error generando recomendaciones: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/agriculture_advice', methods=['POST'])
def get_agriculture_advice():
    """Recomendaciones agrícolas basadas en predicción climática"""
    try:
        data = request.get_json(force=True)
        temperature = data.get('temperature', 20)
        region = data.get('region', 'peru')
        month = data.get('month', 6)
        year = data.get('year', 2030)
        
        advice = {
            'optimal_crops': get_optimal_crops(temperature, region, month),
            'planting_calendar': get_planting_calendar(region, month),
            'irrigation_needs': calculate_irrigation_needs(temperature, region, month),
            'pest_risks': assess_pest_risks(temperature, month),
            'harvest_predictions': get_harvest_predictions(region, month),
            'climate_adaptations': get_climate_adaptation_strategies(temperature, region)
        }
        
        return jsonify({
            'agriculture_advice': advice,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error generando consejos agrícolas: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/extreme_events', methods=['GET'])
def predict_extreme_events():
    """Predicción de eventos climáticos extremos"""
    try:
        year = int(request.args.get('year', 2030))
        region = request.args.get('region', 'peru')
        
        events = analyze_extreme_events(year, region)
        
        return jsonify({
            'extreme_events': events,
            'risk_assessment': calculate_risk_scores(events),
            'preparedness_actions': get_preparedness_actions(events),
            'year': year,
            'region': region,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error prediciendo eventos extremos: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/water_resources', methods=['POST'])
def analyze_water_resources():
    """Análisis de recursos hídricos"""
    try:
        data = request.get_json(force=True)
        temperature = data.get('temperature', 20)
        region = data.get('region', 'peru')
        year = data.get('year', 2030)
        
        analysis = {
            'glacier_melt_rate': estimate_glacier_melt(temperature, region),
            'water_availability': calculate_water_availability(temperature, region, year),
            'drought_risk': assess_drought_risk(temperature, region),
            'flood_risk': assess_flood_risk(temperature, region),
            'recommendations': get_water_management_recommendations(region)
        }
        
        return jsonify({
            'water_analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error analizando recursos hídricos: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# ============ FUNCIONES AUXILIARES NUEVAS ============

def estimate_aqi(temperature, region, month):
    """Estima el índice de calidad del aire"""
    # Base AQI según región (contaminación típica)
    base_aqi = {
        'lima': 75,      # Alta contaminación urbana
        'cusco': 65,     # Media por tráfico
        'arequipa': 60,  # Moderada
        'iquitos': 45,   # Baja (selva)
        'trujillo': 70,  # Moderada-alta
        'peru': 65
    }
    
    aqi = base_aqi.get(region, 65)
    
    # Ajustar por temperatura (inversión térmica)
    if temperature < 15:
        aqi += 10  # Inversión térmica aumenta contaminación
    elif temperature > 28:
        aqi += 5   # Más ozono troposférico
    
    # Ajustar por estación (invierno peor en costa)
    if region in ['lima', 'trujillo'] and 6 <= month <= 9:
        aqi += 15  # Invierno en costa
    
    return min(max(aqi, 0), 300)


def get_aqi_level(aqi):
    """Determina nivel de calidad del aire"""
    if aqi <= 50:
        return 'Bueno'
    elif aqi <= 100:
        return 'Moderado'
    elif aqi <= 150:
        return 'Dañino para grupos sensibles'
    elif aqi <= 200:
        return 'Dañino'
    elif aqi <= 300:
        return 'Muy dañino'
    else:
        return 'Peligroso'


def get_aqi_advice(aqi):
    """Consejos según AQI"""
    if aqi <= 50:
        return "Calidad del aire excelente. Ideal para actividades al aire libre."
    elif aqi <= 100:
        return "Calidad del aire aceptable. Grupos sensibles deben limitar actividad prolongada."
    elif aqi <= 150:
        return "Grupos sensibles (niños, ancianos, asmáticos) deben reducir actividad al aire libre."
    elif aqi <= 200:
        return "Todos deben limitar actividad al aire libre prolongada. Use mascarilla si es necesario."
    else:
        return "EVITE actividades al aire libre. Permanezca en interiores con ventanas cerradas."


def estimate_uv_index(region, month):
    """Estima índice UV según región y mes"""
    # Perú tiene alta radiación UV por ubicación ecuatorial y altitud
    base_uv = {
        'lima': 10,       # Costa, moderado
        'cusco': 14,      # Muy alto por altitud (3400m)
        'arequipa': 13,   # Alto por altitud
        'iquitos': 12,    # Alto por ecuador
        'trujillo': 11,   # Costa norte
        'peru': 11
    }
    
    uv = base_uv.get(region, 11)
    
    # Ajuste estacional (verano más alto)
    if 12 <= month <= 3:  # Verano
        uv += 2
    elif 6 <= month <= 8:  # Invierno
        uv -= 2
    
    return min(max(uv, 1), 16)


def get_uv_level(uv_index):
    """Nivel de riesgo UV"""
    if uv_index <= 2:
        return 'Bajo'
    elif uv_index <= 5:
        return 'Moderado'
    elif uv_index <= 7:
        return 'Alto'
    elif uv_index <= 10:
        return 'Muy alto'
    else:
        return 'Extremo'


def get_uv_advice(uv_index):
    """Consejos de protección UV"""
    if uv_index <= 2:
        return "Mínima protección requerida. Use lentes de sol si hay nieve."
    elif uv_index <= 5:
        return "Use protector solar SPF 30+, sombrero y lentes de sol."
    elif uv_index <= 7:
        return "Protección esencial: SPF 50+, ropa protectora, busque sombra entre 10am-4pm."
    elif uv_index <= 10:
        return "PROTECCIÓN EXTRA: Evite sol entre 10am-4pm, SPF 50+, ropa UV, sombrero de ala ancha."
    else:
        return "RIESGO EXTREMO: Minimice exposición solar. Protección máxima obligatoria."


def get_temperature_health_alerts(temperature, alert_level):
    """Alertas de salud específicas por temperatura"""
    alerts = []
    
    if alert_level in ['extreme_high', 'high']:
        alerts.append({
            'severity': 'high',
            'icon': '🌡️',
            'title': 'Alerta por Calor',
            'message': f'Temperatura de {temperature}°C puede causar estrés térmico',
            'actions': [
                'Manténgase hidratado (2-3 litros de agua/día)',
                'Evite ejercicio intenso entre 11am-4pm',
                'Use ropa ligera y de colores claros',
                'Busque lugares con aire acondicionado o ventilación',
                'Esté atento a síntomas de golpe de calor'
            ]
        })
    
    if alert_level in ['extreme_low', 'low']:
        alerts.append({
            'severity': 'medium',
            'icon': '❄️',
            'title': 'Alerta por Frío',
            'message': f'Temperatura de {temperature}°C requiere precauciones',
            'actions': [
                'Abríguese con varias capas de ropa',
                'Proteja extremidades (manos, pies, orejas)',
                'Evite exposición prolongada al frío',
                'Consuma alimentos calientes y bebidas tibias',
                'Esté atento a síntomas de hipotermia'
            ]
        })
    
    return alerts


def get_general_health_tips(temperature, region, month):
    """Consejos generales de salud"""
    tips = []
    
    # Consejos por región
    if region == 'lima' and 6 <= month <= 9:
        tips.append("Época de inversión térmica en Lima. Riesgo elevado de enfermedades respiratorias.")
    
    if region == 'iquitos':
        tips.append("Zona tropical: Use repelente contra mosquitos (dengue, malaria). Mantenga vacunas al día.")
    
    if region in ['cusco', 'arequipa']:
        tips.append("Altitud elevada: Hidrátese bien, evite alcohol las primeras 48h. Soroche/mal de altura posible.")
    
    # Consejos por temperatura
    if temperature > 25:
        tips.append("Aumente consumo de frutas y verduras frescas ricas en agua.")
    
    if temperature < 15:
        tips.append("Fortalezca sistema inmune: vitamina C, D y alimentos calientes.")
    
    return tips


def get_vulnerable_groups_advice(temperature, alert_level):
    """Consejos para grupos vulnerables"""
    groups = []
    
    if alert_level in ['extreme_high', 'high', 'extreme_low', 'low']:
        groups.append({
            'group': 'Adultos Mayores (+65 años)',
            'risks': 'Mayor susceptibilidad a temperaturas extremas',
            'advice': 'Evite salir en horas críticas. Mantenga contacto frecuente con familiares.'
        })
        
        groups.append({
            'group': 'Niños menores de 5 años',
            'risks': 'Sistema de termorregulación inmaduro',
            'advice': 'Supervise hidratación. Vístales apropiadamente. Evite exposición prolongada.'
        })
        
        groups.append({
            'group': 'Personas con enfermedades crónicas',
            'risks': 'Cardiovasculares, respiratorias, diabetes',
            'advice': 'Mantenga medicamentos a mano. Consulte síntomas inusuales inmediatamente.'
        })
        
        groups.append({
            'group': 'Mujeres embarazadas',
            'risks': 'Mayor riesgo de deshidratación y estrés térmico',
            'advice': 'Hidratación constante. Evite esfuerzos. Consulte cualquier malestar.'
        })
    
    return groups


def get_optimal_crops(temperature, region, month):
    """Cultivos óptimos según condiciones"""
    crops = []
    
    # Cultivos por región y temperatura
    if region in ['lima', 'trujillo'] and 18 <= temperature <= 26:
        crops = [
            {'name': 'Espárrago', 'season': 'Todo el año', 'water': 'Media', 'yield': 'Alta'},
            {'name': 'Maíz', 'season': 'Sep-Mar', 'water': 'Media', 'yield': 'Alta'},
            {'name': 'Algodón', 'season': 'Ago-Mar', 'water': 'Media-Alta', 'yield': 'Media'},
            {'name': 'Tomate', 'season': 'Todo el año', 'water': 'Media', 'yield': 'Alta'}
        ]
    
    elif region in ['cusco', 'arequipa'] and 10 <= temperature <= 20:
        crops = [
            {'name': 'Papa', 'season': 'Sep-May', 'water': 'Media', 'yield': 'Muy Alta'},
            {'name': 'Quinua', 'season': 'Sep-May', 'water': 'Baja', 'yield': 'Alta'},
            {'name': 'Cebada', 'season': 'Oct-Abr', 'water': 'Baja', 'yield': 'Media'},
            {'name': 'Habas', 'season': 'Ago-Abr', 'water': 'Media', 'yield': 'Alta'}
        ]
    
    elif region == 'iquitos' and temperature >= 24:
        crops = [
            {'name': 'Cacao', 'season': 'Todo el año', 'water': 'Alta', 'yield': 'Alta'},
            {'name': 'Plátano', 'season': 'Todo el año', 'water': 'Alta', 'yield': 'Muy Alta'},
            {'name': 'Yuca', 'season': 'Todo el año', 'water': 'Media', 'yield': 'Alta'},
            {'name': 'Café', 'season': 'Abr-Sep', 'water': 'Media-Alta', 'yield': 'Alta'}
        ]
    
    return crops


def get_planting_calendar(region, current_month):
    """Calendario de siembra recomendado"""
    months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    
    calendar = {
        'lima': {
            'siembra': [9, 10, 11],  # Sep-Nov
            'cultivos': ['Maíz', 'Frijol', 'Tomate'],
            'preparacion': [7, 8],  # Jul-Ago
            'cosecha': [3, 4, 5]  # Mar-May
        },
        'cusco': {
            'siembra': [10, 11, 12],  # Oct-Dic
            'cultivos': ['Papa', 'Quinua', 'Maíz'],
            'preparacion': [8, 9],
            'cosecha': [4, 5, 6]
        },
        'iquitos': {
            'siembra': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Todo el año
            'cultivos': ['Yuca', 'Plátano', 'Cacao'],
            'preparacion': [],
            'cosecha': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        }
    }
    
    region_cal = calendar.get(region, calendar['lima'])
    
    status = 'siembra' if current_month in region_cal['siembra'] else \
             'preparacion' if current_month in region_cal['preparacion'] else \
             'cosecha' if current_month in region_cal['cosecha'] else 'mantenimiento'
    
    return {
        'current_status': status,
        'current_month_name': months[current_month - 1],
        'optimal_siembra': [months[m-1] for m in region_cal['siembra']],
        'optimal_cosecha': [months[m-1] for m in region_cal['cosecha']],
        'recommended_crops': region_cal['cultivos']
    }


def calculate_irrigation_needs(temperature, region, month):
    """Calcula necesidades de irrigación"""
    # Base de riego (mm/día)
    base_needs = 4
    
    # Ajuste por temperatura
    if temperature > 28:
        base_needs += 2
    elif temperature > 24:
        base_needs += 1
    elif temperature < 15:
        base_needs -= 1
    
    # Ajuste por región (evapotranspiración)
    if region in ['iquitos']:
        base_needs += 1  # Más evaporación en selva
    elif region in ['cusco', 'arequipa']:
        base_needs += 0.5  # Aire seco en altura
    
    # Ajuste estacional (verano más riego)
    if 12 <= month <= 3:
        base_needs += 1
    
    return {
        'daily_mm': round(base_needs, 1),
        'weekly_mm': round(base_needs * 7, 1),
        'frequency': 'Diario' if base_needs > 6 else 'Cada 2-3 días',
        'method': 'Goteo' if region in ['lima', 'arequipa'] else 'Aspersión',
        'conservation_tips': [
            'Use riego por goteo para 50% menos agua',
            'Riegue temprano (5-7am) o tarde (6-8pm)',
            'Implemente mulch para retener humedad',
            'Monitoree humedad del suelo antes de regar'
        ]
    }


def assess_pest_risks(temperature, month):
    """Evalúa riesgos de plagas según temperatura"""
    risks = []
    
    if temperature > 22:
        risks.append({
            'pest': 'Mosca de la fruta',
            'risk_level': 'Alto',
            'crops_affected': ['Mango', 'Naranja', 'Mandarina'],
            'prevention': 'Trampas atrayentes, control biológico'
        })
    
    if 20 <= temperature <= 28:
        risks.append({
            'pest': 'Pulgones',
            'risk_level': 'Medio',
            'crops_affected': ['Tomate', 'Pimiento', 'Lechuga'],
            'prevention': 'Insecticidas naturales (ajo, nim)'
        })
    
    if temperature > 25 and month in [12, 1, 2, 3]:
        risks.append({
            'pest': 'Gusano cogollero',
            'risk_level': 'Alto',
            'crops_affected': ['Maíz'],
            'prevention': 'Monitoreo semanal, Bacillus thuringiensis'
        })
    
    return risks


def get_harvest_predictions(region, month):
    """Predicciones de cosecha"""
    predictions = {
        'estimated_yield': 'Normal',
        'quality': 'Buena',
        'market_timing': 'Favorable',
        'storage_conditions': 'Óptimas'
    }
    
    if region in ['cusco', 'arequipa'] and 4 <= month <= 6:
        predictions['estimated_yield'] = 'Alta'
        predictions['quality'] = 'Excelente'
        predictions['recommendation'] = 'Época óptima para cosecha de papa y quinua'
    
    return predictions


def get_climate_adaptation_strategies(temperature, region):
    """Estrategias de adaptación al cambio climático"""
    strategies = []
    
    if temperature > 25:
        strategies.append({
            'strategy': 'Variedades termo-tolerantes',
            'description': 'Usar semillas adaptadas a mayor calor',
            'implementation': 'Contactar INIA para variedades mejoradas'
        })
    
    strategies.append({
        'strategy': 'Agro-forestación',
        'description': 'Plantar árboles para sombra y regulación térmica',
        'implementation': 'Árboles nativos en bordes de parcelas'
    })
    
    strategies.append({
        'strategy': 'Cosecha de agua',
        'description': 'Sistemas de captación de lluvia y qochas',
        'implementation': 'Reservorios, zanjas de infiltración'
    })
    
    return strategies


def analyze_extreme_events(year, region):
    """Analiza probabilidad de eventos extremos"""
    years_ahead = year - 2024
    
    # Probabilidad aumenta con años
    base_prob = min(0.3 + years_ahead * 0.02, 0.7)
    
    events = []
    
    # Olas de calor
    if region in ['lima', 'trujillo', 'iquitos']:
        events.append({
            'event': 'Ola de calor',
            'probability': f"{int(base_prob * 100)}%",
            'severity': 'Media-Alta',
            'duration': '5-10 días',
            'impacts': ['Estrés hídrico en cultivos', 'Riesgo de incendios', 'Problemas de salud']
        })
    
    # Heladas
    if region in ['cusco', 'arequipa']:
        events.append({
            'event': 'Heladas intensas',
            'probability': f"{int((base_prob + 0.1) * 100)}%",
            'severity': 'Alta',
            'duration': 'Jun-Ago',
            'impacts': ['Pérdida de cultivos andinos', 'Daño a ganado', 'Riesgo de hipotermia']
        })
    
    # Sequías
    events.append({
        'event': 'Sequía prolongada',
        'probability': f"{int((base_prob + 0.05) * 100)}%",
        'severity': 'Alta',
        'duration': '3-6 meses',
        'impacts': ['Escasez de agua', 'Reducción de cosechas', 'Migración rural']
    })
    
    # El Niño/La Niña
    events.append({
        'event': 'Fenómeno El Niño',
        'probability': f"{int(0.4 * 100)}%",
        'severity': 'Variable',
        'duration': '6-12 meses',
        'impacts': ['Lluvias intensas en costa', 'Sequía en sierra sur', 'Pérdidas económicas']
    })
    
    return events


def calculate_risk_scores(events):
    """Calcula scores de riesgo"""
    severity_map = {'Baja': 1, 'Media': 2, 'Media-Alta': 3, 'Alta': 4, 'Muy Alta': 5}
    
    scores = []
    for event in events:
        prob_value = int(event['probability'].replace('%', '')) / 100
        severity_value = severity_map.get(event['severity'].split('-')[0], 3)
        risk_score = prob_value * severity_value
        
        scores.append({
            'event': event['event'],
            'risk_score': round(risk_score, 2),
            'risk_level': 'Crítico' if risk_score > 3 else 'Alto' if risk_score > 2 else 'Moderado'
        })
    
    return scores


def get_preparedness_actions(events):
    """Acciones de preparación para eventos extremos"""
    actions = {
        'immediate': [
            'Crear plan de emergencia familiar',
            'Identificar refugios cercanos',
            'Preparar kit de emergencia (agua, alimentos, medicinas)'
        ],
        'short_term': [
            'Reforzar infraestructura vulnerable',
            'Diversificar cultivos',
            'Implementar sistemas de alerta temprana'
        ],
        'long_term': [
            'Invertir en infraestructura resiliente',
            'Capacitación en gestión de riesgos',
            'Seguro agrícola y de salud'
        ]
    }
    
    return actions


def estimate_glacier_melt(temperature, region):
    """Estima tasa de derretimiento de glaciares"""
    if region not in ['cusco', 'arequipa', 'peru']:
        return {'applicable': False, 'message': 'Sin glaciares significativos en esta región'}
    
    # Temperatura base óptima para glaciares: <2°C
    baseline_temp = 2
    excess_temp = temperature - baseline_temp
    
    if excess_temp <= 0:
        melt_rate = 0.5  # Mínima pérdida
    else:
        melt_rate = 1.5 + (excess_temp * 0.3)  # Aumenta con temperatura
    
    return {
        'applicable': True,
        'annual_loss_percent': round(melt_rate, 1),
        'status': 'Crítico' if melt_rate > 2 else 'Preocupante' if melt_rate > 1 else 'Estable',
        'impacts': [
            'Reducción de agua en época seca',
            'Afecta agricultura y ciudades de sierra',
            'Pérdida de reservas hídricas a largo plazo'
        ],
        'affected_glaciers': [
            'Cordillera Blanca (Áncash)',
            'Nevado Coropuna (Arequipa)',
            'Ausangate (Cusco)'
        ]
    }


def calculate_water_availability(temperature, region, year):
    """Calcula disponibilidad de agua proyectada"""
    base_availability = {
        'lima': 650,  # m³ per cápita/año (escasez)
        'cusco': 1200,
        'arequipa': 800,
        'iquitos': 35000,  # Abundante
        'trujillo': 700,
        'peru': 1800
    }
    
    availability = base_availability.get(region, 1800)
    
    # Reducción por calentamiento (glaciares, evaporación)
    years_ahead = year - 2024
    reduction_rate = 0.015  # 1.5% por año
    availability *= (1 - reduction_rate) ** years_ahead
    
    # Clasificación según ONU
    if availability < 1000:
        status = 'Escasez crítica'
        alert = 'high'
    elif availability < 1700:
        status = 'Estrés hídrico'
        alert = 'medium'
    else:
        status = 'Suficiente'
        alert = 'low'
    
    return {
        'per_capita_m3': round(availability, 0),
        'status': status,
        'alert_level': alert,
        'trend': f"-{round(reduction_rate * 100, 1)}% anual",
        'recommendations': get_water_management_recommendations(region)
    }


def assess_drought_risk(temperature, region):
    """Evalúa riesgo de sequía"""
    base_risk = 0.3
    
    # Aumenta con temperatura
    if temperature > 26:
        base_risk += 0.2
    elif temperature > 24:
        base_risk += 0.1
    
    # Regiones más vulnerables
    if region in ['arequipa', 'trujillo']:
        base_risk += 0.15
    
    risk_level = 'Alto' if base_risk > 0.6 else 'Medio' if base_risk > 0.4 else 'Bajo'
    
    return {
        'risk_percentage': f"{int(base_risk * 100)}%",
        'risk_level': risk_level,
        'vulnerable_sectors': ['Agricultura', 'Ganadería', 'Generación hidroeléctrica'],
        'early_warning_signs': [
            'Reducción de caudales en ríos',
            'Niveles bajos en reservorios',
            'Estrés en vegetación'
        ]
    }


def assess_flood_risk(temperature, region):
    """Evalúa riesgo de inundación"""
    base_risk = 0.25
    
    # Mayor temperatura = más evaporación = lluvias más intensas
    if temperature > 28:
        base_risk += 0.25
    
    # Regiones costeras más vulnerables a El Niño
    if region in ['lima', 'trujillo', 'iquitos']:
        base_risk += 0.20
    
    risk_level = 'Alto' if base_risk > 0.6 else 'Medio' if base_risk > 0.35 else 'Bajo'
    
    return {
        'risk_percentage': f"{int(base_risk * 100)}%",
        'risk_level': risk_level,
        'vulnerable_areas': [
            'Zonas bajas cerca de ríos',
            'Cauces secos (huaicos)',
            'Áreas sin drenaje adecuado'
        ],
        'prevention_measures': [
            'Limpieza de canales y alcantarillas',
            'Muros de contención en quebradas',
            'Sistema de alerta temprana',
            'Planes de evacuación actualizados'
        ]
    }


def get_water_management_recommendations(region):
    """Recomendaciones de gestión hídrica"""
    recommendations = [
        {
            'category': 'Conservación',
            'actions': [
                'Instalar sistemas de riego tecnificado',
                'Reúso de aguas grises tratadas',
                'Captar agua de lluvia en techos y reservorios',
                'Reducir pérdidas en redes de distribución'
            ]
        },
        {
            'category': 'Adaptación',
            'actions': [
                'Diversificar fuentes de agua',
                'Construir reservorios estratégicos',
                'Recarga artificial de acuíferos',
                'Desalinización (zonas costeras)'
            ]
        },
        {
            'category': 'Gobernanza',
            'actions': [
                'Fortalecer juntas de usuarios',
                'Tarifas progresivas por consumo',
                'Monitoreo en tiempo real',
                'Educación en uso responsable'
            ]
        }
    ]
    
    return recommendations


@app.route('/biodiversity_impact', methods=['POST'])
def assess_biodiversity_impact():
    """Evaluación de impacto en biodiversidad"""
    try:
        data = request.get_json(force=True)
        temperature = data.get('temperature', 20)
        region = data.get('region', 'peru')
        year = data.get('year', 2030)
        
        impact = {
            'species_at_risk': get_threatened_species(temperature, region),
            'habitat_loss': estimate_habitat_loss(temperature, region, year),
            'migration_patterns': assess_species_migration(temperature, region),
            'ecosystem_services': evaluate_ecosystem_services(temperature, region),
            'conservation_priorities': get_conservation_priorities(region)
        }
        
        return jsonify({
            'biodiversity_impact': impact,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error evaluando biodiversidad: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


def get_threatened_species(temperature, region):
    """Especies amenazadas por cambio climático"""
    species = {
        'cusco': [
            {'name': 'Oso de anteojos', 'threat_level': 'Alto', 'reason': 'Pérdida de hábitat'},
            {'name': 'Colibrí maravilloso', 'threat_level': 'Crítico', 'reason': 'Rango limitado'},
            {'name': 'Taruca', 'threat_level': 'Alto', 'reason': 'Cambios en vegetación'}
        ],
        'iquitos': [
            {'name': 'Jaguar', 'threat_level': 'Medio', 'reason': 'Fragmentación de selva'},
            {'name': 'Delfín rosado', 'threat_level': 'Alto', 'reason': 'Cambios en ríos'},
            {'name': 'Guacamayo rojo', 'threat_level': 'Medio', 'reason': 'Pérdida de palmas'}
        ],
        'lima': [
            {'name': 'Pingüino de Humboldt', 'threat_level': 'Alto', 'reason': 'Temperatura del mar'},
            {'name': 'Lobo marino', 'threat_level': 'Medio', 'reason': 'Cambios en pesca'},
            {'name': 'Potoyunco', 'threat_level': 'Alto', 'reason': 'Hábitat costero reducido'}
        ]
    }
    
    return species.get(region, species['lima'])


def estimate_habitat_loss(temperature, region, year):
    """Estima pérdida de hábitat"""
    years_ahead = year - 2024
    base_loss = 0.02  # 2% anual
    
    if temperature > 25:
        base_loss += 0.01
    
    total_loss = min(base_loss * years_ahead * 100, 50)
    
    return {
        'percentage_loss': round(total_loss, 1),
        'primary_causes': [
            'Expansión agrícola por migración de cultivos',
            'Aumento de incendios forestales',
            'Desertificación en zonas áridas',
            'Blanqueamiento de corales (costa)'
        ],
        'most_affected': [
            'Bosques montanos (2500-3500m)',
            'Humedales costeros',
            'Bosques secos del norte'
        ]
    }


def assess_species_migration(temperature, region):
    """Evalúa migración de especies"""
    return {
        'altitudinal_shift': f"+{int((temperature - 18) * 50)}m por década",
        'direction': 'Hacia mayor altitud y latitudes más altas',
        'impacts': [
            'Competencia con especies nativas en nuevas áreas',
            'Desaparición de especies sin lugar para migrar',
            'Cambios en interacciones depredador-presa'
        ],
        'monitoring_needed': True
    }


def evaluate_ecosystem_services(temperature, region):
    """Evalúa servicios ecosistémicos"""
    services = [
        {
            'service': 'Provisión de agua',
            'status': 'Degradado',
            'impact': 'Reducción 15-30% para 2050'
        },
        {
            'service': 'Polinización',
            'status': 'En riesgo',
            'impact': 'Pérdida de polinizadores afecta agricultura'
        },
        {
            'service': 'Control de erosión',
            'status': 'Crítico',
            'impact': 'Aumento de deslizamientos y huaicos'
        },
        {
            'service': 'Captura de carbono',
            'status': 'Reducido',
            'impact': 'Bosques bajo estrés capturan menos CO2'
        }
    ]
    
    return services


def get_conservation_priorities(region):
    """Prioridades de conservación"""
    return {
        'immediate_actions': [
            'Expandir áreas naturales protegidas',
            'Corredores ecológicos entre hábitats fragmentados',
            'Restauración de ecosistemas degradados',
            'Programas de reproducción ex-situ para especies críticas'
        ],
        'long_term_strategies': [
            'Conservación basada en comunidades',
            'Pago por servicios ecosistémicos',
            'Investigación y monitoreo continuo',
            'Educación ambiental masiva'
        ],
        'funding_sources': [
            'Fondo Nacional de Áreas Naturales Protegidas',
            'Cooperación internacional (GEF, GCF)',
            'Bonos de carbono y compensación ambiental',
            'Ecoturismo sostenible'
        ]
    }


@app.route('/economic_impact', methods=['POST'])
def calculate_economic_impact():
    """Calcula impacto económico del cambio climático"""
    try:
        data = request.get_json(force=True)
        temperature = data.get('temperature', 20)
        region = data.get('region', 'peru')
        year = data.get('year', 2030)
        
        impact = {
            'gdp_impact': estimate_gdp_impact(temperature, year),
            'sector_impacts': calculate_sector_impacts(temperature, region),
            'infrastructure_costs': estimate_infrastructure_costs(region, year),
            'adaptation_investment': calculate_adaptation_needs(region),
            'job_impacts': assess_employment_effects(temperature, region)
        }
        
        return jsonify({
            'economic_impact': impact,
            'currency': 'USD millones',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error calculando impacto económico: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


def estimate_gdp_impact(temperature, year):
    """Estima impacto en PIB"""
    years_ahead = year - 2024
    baseline_loss = 0.015  # 1.5% PIB por año sin acción
    
    cumulative_loss = baseline_loss * years_ahead
    total_loss_pct = min(cumulative_loss * 100, 20)
    
    peru_gdp_2024 = 240000  # USD millones aprox
    loss_amount = peru_gdp_2024 * (total_loss_pct / 100)
    
    return {
        'percentage_loss': round(total_loss_pct, 1),
        'estimated_loss_usd_millions': round(loss_amount, 0),
        'per_capita_impact_usd': round(loss_amount * 1000000 / 34000000, 0),
        'comparison': f'Equivalente a {round(loss_amount/peru_gdp_2024*100, 1)}% del PIB actual'
    }


def calculate_sector_impacts(temperature, region):
    """Impactos por sector económico"""
    sectors = [
        {
            'sector': 'Agricultura',
            'impact_level': 'Muy Alto',
            'loss_percentage': 20,
            'description': 'Reducción de rendimientos, cambio de cultivos'
        },
        {
            'sector': 'Pesca',
            'impact_level': 'Alto',
            'loss_percentage': 15,
            'description': 'Migración de especies, acidificación oceánica'
        },
        {
            'sector': 'Turismo',
            'impact_level': 'Medio-Alto',
            'loss_percentage': 12,
            'description': 'Pérdida de glaciares, daño a ecosistemas'
        },
        {
            'sector': 'Energía',
            'impact_level': 'Alto',
            'loss_percentage': 18,
            'description': 'Menor caudal en hidroeléctricas (70% de energía)'
        },
        {
            'sector': 'Infraestructura',
            'impact_level': 'Muy Alto',
            'loss_percentage': 25,
            'description': 'Daños por eventos extremos, adaptación necesaria'
        }
    ]
    
    return sectors


def estimate_infrastructure_costs(region, year):
    """Estima costos de infraestructura"""
    years_ahead = year - 2024
    annual_cost = 500  # USD millones/año
    
    total_cost = annual_cost * years_ahead
    
    return {
        'total_investment_needed_usd_millions': round(total_cost, 0),
        'breakdown': {
            'flood_defense': round(total_cost * 0.30, 0),
            'water_infrastructure': round(total_cost * 0.25, 0),
            'transportation': round(total_cost * 0.20, 0),
            'health_facilities': round(total_cost * 0.15, 0),
            'early_warning_systems': round(total_cost * 0.10, 0)
        },
        'financing_gap': 'Se requiere aumentar inversión en 3-4x'
    }


def calculate_adaptation_needs(region):
    """Necesidades de inversión en adaptación"""
    return {
        'total_needed_usd_millions': 10000,
        'timeframe': '2024-2030',
        'priority_areas': [
            {
                'area': 'Agricultura climáticamente inteligente',
                'investment': 2500,
                'expected_benefit': 'Mantener producción alimentaria'
            },
            {
                'area': 'Infraestructura hídrica',
                'investment': 3000,
                'expected_benefit': 'Seguridad hídrica para 80% población'
            },
            {
                'area': 'Protección costera',
                'investment': 2000,
                'expected_benefit': 'Proteger 60% de costa habitada'
            },
            {
                'area': 'Sistemas de alerta temprana',
                'investment': 1500,
                'expected_benefit': 'Reducir muertes por eventos extremos 90%'
            },
            {
                'area': 'Restauración de ecosistemas',
                'investment': 1000,
                'expected_benefit': 'Servicios ecosistémicos valorados en $5B'
            }
        ]
    }


def assess_employment_effects(temperature, region):
    """Efectos en empleo"""
    return {
        'jobs_at_risk': 850000,
        'most_affected_sectors': [
            'Agricultura (500,000 empleos)',
            'Pesca (150,000 empleos)',
            'Turismo (200,000 empleos)'
        ],
        'new_opportunities': [
            'Energías renovables (+100,000 empleos)',
            'Agricultura climáticamente inteligente (+80,000)',
            'Gestión de recursos hídricos (+40,000)',
            'Servicios ambientales (+30,000)'
        ],
        'transition_support': [
            'Programas de capacitación',
            'Subsidios para reconversión',
            'Apoyo a emprendimientos verdes'
        ]
    }

@app.route('/download_predictions', methods=['POST'])
def download_predictions():
    """Genera CSV detallado de predicciones para descarga"""
    try:
        data = request.get_json(force=True)
        
        # Parámetros de configuración
        year = data.get('year', 2030)
        model_name = data.get('model', 'neural_net')
        regions = data.get('regions', ['peru', 'lima', 'cusco', 'arequipa', 'iquitos', 'trujillo'])
        months = data.get('months', list(range(1, 13)))  # Todos los meses por defecto
        
        # Validar parámetros
        if year < 2024 or year > 2100:
            return jsonify({
                'error': f'Año {year} fuera de rango (2024-2100)',
                'status': 'error'
            }), 400
        
        if model_name not in climate_predictor.models:
            return jsonify({
                'error': f'Modelo {model_name} no disponible',
                'status': 'error'
            }), 400
        
        # Generar predicciones
        predictions = climate_predictor.generate_predictions_csv(
            year, model_name, regions, months
        )
        
        # Convertir a DataFrame y CSV
        df = pd.DataFrame(predictions)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_content = csv_buffer.getvalue()
        
        # Generar estadísticas del archivo
        file_stats = {
            'total_predictions': len(predictions),
            'regions_included': len(set(df['region'])),
            'months_included': len(set(df['month'])),
            'avg_temperature': round(df['predicted_temperature'].mean(), 1),
            'temperature_range': {
                'min': round(df['predicted_temperature'].min(), 1),
                'max': round(df['predicted_temperature'].max(), 1)
            },
            'avg_confidence': round(df['confidence_percent'].mean(), 1),
            'alert_distribution': df['alert_level'].value_counts().to_dict()
        }
        
        return jsonify({
            'csv_content': csv_content,
            'filename': f'predicciones_clima_peru_{year}_{model_name}.csv',
            'file_stats': file_stats,
            'generation_info': {
                'model_used': model_name,
                'year': year,
                'regions': regions,
                'months_count': len(months),
                'generated_at': datetime.now().isoformat()
            },
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error generando CSV: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/comparison/<regions>', methods=['GET'])
def compare_regions(regions):
    """Comparación entre múltiples regiones"""
    try:
        region_list = [r.strip().lower() for r in regions.split(',')]
        year = int(request.args.get('year', 2030))
        month = int(request.args.get('month', 6))
        model_name = request.args.get('model', 'neural_net')
        
        comparison_data = []
        
        for region in region_list:
            # Datos históricos
            historical = climate_predictor.get_historical_data(region)
            
            # Predicción futura
            coords = {
                'lima': {'lat': -12.0464, 'lng': -77.0428, 'alt': 154},
                'cusco': {'lat': -13.5319, 'lng': -71.9675, 'alt': 3399},
                'arequipa': {'lat': -16.4090, 'lng': -71.5375, 'alt': 2335},
                'iquitos': {'lat': -3.7437, 'lng': -73.2516, 'alt': 106},
                'trujillo': {'lat': -8.1116, 'lng': -79.0290, 'alt': 34},
                'peru': {'lat': -9.1900, 'lng': -75.0152, 'alt': 500}
            }.get(region, {'lat': -12, 'lng': -75, 'alt': 500})
            
            features = {
                'year': year, 'year_since_1990': year - 1990, 'month': month,
                'season': ((month - 1) // 3) + 1, **coords,
                'altitude_norm': (coords['alt'] - 1500) / 1500,
                'coastal': 1 if region in ['lima', 'trujillo'] else 0,
                'highland': 1 if region in ['cusco', 'arequipa'] else 0,
                'jungle': 1 if region == 'iquitos' else 0,
                'el_nino_index': np.sin(2 * np.pi * (year - 1990) / 4.5),
                'solar_cycle': np.sin(2 * np.pi * (year - 1990) / 11),
                'linear_trend': year - 1990,
                'quadratic_trend': (year - 1990) ** 2,
                'decade': (year - 1990) // 10
            }
            
            temp, confidence = climate_predictor.predict_temperature(
                features, model_name, region
            )
            
            comparison_data.append({
                'region': region,
                'region_name': region.replace('_', ' ').title(),
                'historical_data': historical,
                'prediction': {
                    'temperature': round(temp, 1),
                    'confidence': round(confidence, 1),
                    'year': year,
                    'month': month
                },
                'statistics': {
                    'historical_avg': round(np.mean(historical['temperatures']), 1),
                    'warming_trend': calculate_temperature_trend(
                        historical['years'], historical['temperatures']
                    ),
                    'vs_prediction': round(temp - np.mean(historical['temperatures']), 1)
                }
            })
        
        return jsonify({
            'regions': comparison_data,
            'comparison_summary': {
                'warmest_region': max(comparison_data, key=lambda x: x['prediction']['temperature'])['region'],
                'coolest_region': min(comparison_data, key=lambda x: x['prediction']['temperature'])['region'],
                'avg_temperature': round(np.mean([r['prediction']['temperature'] for r in comparison_data]), 1),
                'temperature_range': round(
                    max([r['prediction']['temperature'] for r in comparison_data]) - 
                    min([r['prediction']['temperature'] for r in comparison_data]), 1
                )
            },
            'parameters': {'year': year, 'month': month, 'model': model_name},
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error en comparación: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


# Servir archivos estáticos
@app.route('/static/<path:filename>')
def static_files(filename):
    """Servir archivos estáticos con manejo mejorado"""
    try:
        static_dirs = [
            'static',
            'frontend/static',
            '../frontend/static',
            'assets',
            'frontend/assets'
        ]
        
        for static_dir in static_dirs:
            if os.path.exists(static_dir):
                full_path = os.path.join(static_dir, filename)
                if os.path.exists(full_path):
                    return send_from_directory(static_dir, filename)
        
        logger.warning(f"Archivo estático no encontrado: {filename}")
        return "Archivo no encontrado", 404
        
    except Exception as e:
        logger.error(f"Error sirviendo archivo estático: {e}")
        return "Error interno del servidor", 500


# ===============================
# FUNCIONES AUXILIARES
# ===============================

def get_climate_context(temperature, region):
    """Proporciona contexto climático detallado con 10 niveles para cada región"""
    
    # Base de datos de temperaturas extremas históricas por región
    historical_records = {
        'lima': {
            'min_temp': 8.2,    # Temperatura mínima histórica registrada
            'max_temp': 34.2,   # Temperatura máxima histórica registrada
            'avg_temp': 19.0,   # Temperatura promedio anual
            'extreme_events': [
                {'year': 1998, 'event': 'Ola de calor', 'temp': 34.2},
                {'year': 2017, 'event': 'Lluvias intensas', 'temp': 15.3},
                {'year': 2023, 'event': 'Ola de calor', 'temp': 33.5}
            ]
        },
        'cusco': {
            'min_temp': -2.0,
            'max_temp': 28.5,
            'avg_temp': 15.0,
            'extreme_events': [
                {'year': 2003, 'event': 'Heladas severas', 'temp': -2.0},
                {'year': 2013, 'event': 'Heladas', 'temp': -1.5},
                {'year': 2021, 'event': 'Lluvias torrenciales', 'temp': 20.3}
            ]
        },
        'arequipa': {
            'min_temp': 1.5,
            'max_temp': 32.8,
            'avg_temp': 18.0,
            'extreme_events': [
                {'year': 2007, 'event': 'Heladas', 'temp': 1.5},
                {'year': 2015, 'event': 'Sequía', 'temp': 25.4},
                {'year': 2019, 'event': 'Lluvias intensas', 'temp': 22.7}
            ]
        },
        'iquitos': {
            'min_temp': 16.8,
            'max_temp': 38.5,
            'avg_temp': 26.0,
            'extreme_events': [
                {'year': 2012, 'event': 'Inundaciones', 'temp': 24.3},
                {'year': 2015, 'event': 'Inundaciones', 'temp': 25.1},
                {'year': 2020, 'event': 'Nivel récord de ríos', 'temp': 27.8}
            ]
        },
        'trujillo': {
            'min_temp': 12.5,
            'max_temp': 35.8,
            'avg_temp': 22.0,
            'extreme_events': [
                {'year': 2014, 'event': 'Sequía', 'temp': 28.7},
                {'year': 2016, 'event': 'Ola de calor', 'temp': 35.8},
                {'year': 2021, 'event': 'Temperatura extrema', 'temp': 34.9}
            ]
        }
    }
    
    # Contextos detallados con 10 niveles para cada región
    contexts = {
        'lima': {
            (0, 10): {
                'context': "Temperatura extremadamente fría para Lima - Evento sin precedentes",
                'severity': "EXTREMO",
                'recommendations': [
                    "Evitar cualquier exposición al frío extremo",
                    "Proteger sistemas de agua para evitar congelamiento",
                    "Abrigar plantas tropicales y cultivos sensibles",
                    "Precaución en carreteras de montaña",
                    "Monitorear a personas mayores y niños"
                ]
            },
            (10, 14): {
                'context': "Temperatura muy fría para Lima - Evento muy inusual",
                'severity': "ALTO",
                'recommendations': [
                    "Usar abrigos pesados y gorros",
                    "Proteger tuberías con materiales aislantes",
                    "Evitar actividades al aire libre prolongadas",
                    "Calefaccionar espacios interiores",
                    "Proteger mascotas y animales domésticos"
                ]
            },
            (14, 18): {
                'context': "Temperatura fría atípica para Lima - Influencia de aire antártico",
                'severity': "MODERADO",
                'recommendations': [
                    "Vestir en capas con ropa de abrigo",
                    "Proteger cultivos sensibles con cubiertas",
                    "Conducir con precaución por posible neblina",
                    "Mantener ventilación moderada en interiores",
                    "Evitar cambios bruscos de temperatura"
                ]
            },
            (18, 22): {
                'context': "Temperatura fresca típica de invierno limeño - Condiciones normales",
                'severity': "NORMAL",
                'recommendations': [
                    "Condiciones ideales para actividades diarias",
                    "Buen momento para caminatas y ejercicio",
                    "Aprovechar para turismo en la ciudad",
                    "Ventilar espacios durante las horas más cálidas",
                    "Mantener rutinas normales de cuidado personal"
                ]
            },
            (22, 26): {
                'context': "Temperatura agradable para Lima - Condiciones ideales",
                'severity': "NORMAL",
                'recommendations': [
                    "Excelente para actividades al aire libre",
                    "Ideal para deportes y recreación",
                    "Buen momento para eventos sociales",
                    "Aprovechar para fotografía urbana",
                    "Disfrutar de parques y áreas verdes"
                ]
            },
            (26, 30): {
                'context': "Temperatura cálida para Lima - Comienzo de condiciones de calor",
                'severity': "MODERADO",
                'recommendations': [
                    "Mantenerse hidratado constantemente",
                    "Protegerse del sol entre 11am-3pm",
                    "Usar ropa ligera y de colores claros",
                    "Ventilar espacios cerrados",
                    "Evitar ejercicio físico intenso"
                ]
            },
            (30, 34): {
                'context': "Temperatura muy alta para Lima - Posible ola de calor",
                'severity': "ALTO",
                'recommendations': [
                    "Evitar actividades físicas intensas",
                    "Permanecer en lugares con aire acondicionado",
                    "Proteger a grupos vulnerables (niños, ancianos)",
                    "Aumentar ingesta de líquidos y electrolitos",
                    "Monitorear síntomas de agotamiento por calor"
                ]
            },
            (34, 38): {
                'context': "Temperatura extrema para Lima - Ola de calor severa",
                'severity': "EXTREMO",
                'recommendations': [
                    "Evitar salir en horas pico de calor (10am-6pm)",
                    "Buscar refugios climáticos con aire acondicionado",
                    "Seguir recomendaciones de autoridades sanitarias",
                    "Protegerse con ropa de manga larga y sombrero",
                    "Vigilar signos de golpe de calor"
                ]
            },
            (38, 42): {
                'context': "Temperatura récord para Lima - Evento histórico sin precedentes",
                'severity': "CRÍTICO",
                'recommendations': [
                    "Permanecer obligatoriamente en interiores",
                    "Activar planes de emergencia por calor extremo",
                    "Monitorear constantemente a personas vulnerables",
                    "Seguir instrucciones de defensa civil",
                    "Preparar para posibles cortes de energía"
                ]
            },
            (42, 50): {
                'context': "Temperatura extrema sin precedentes - Situación de emergencia nacional",
                'severity': "EMERGENCIA",
                'recommendations': [
                    "Declarar estado de emergencia por calor extremo",
                    "Activar todos los sistemas de respuesta civil",
                    "Evitar cualquier exposición al calor extremo",
                    "Proteger infraestructura crítica",
                    "Seguir protocolos de emergencia nacional"
                ]
            }
        },
        'cusco': {
            (0, 5): {
                'context': "Temperatura extremadamente fría para Cusco - Riesgo de heladas severas",
                'severity': "EXTREMO",
                'recommendations': [
                    "Proteger cultivos con cubiertas térmicas especiales",
                    "Evitar exposición nocturna en zonas altas",
                    "Proteger sistemas de agua del congelamiento",
                    "Abrigar ganado y animales domésticos",
                    "Activar alertas por heladas en comunidades"
                ]
            },
            (5, 10): {
                'context': "Temperatura muy fría para Cusco - Probables heladas nocturnas",
                'severity': "ALTO",
                'recommendations': [
                    "Usar varias capas de ropa térmica",
                    "Proteger tuberías con materiales aislantes",
                    "Cuidado especial con personas mayores y niños",
                    "Evitar viajes a zonas de alta montaña",
                    "Proteger cultivos andinos sensibles"
                ]
            },
            (10, 15): {
                'context': "Temperatura fría típica de Cusco en época seca",
                'severity': "MODERADO",
                'recommendations': [
                    "Usar abrigos especialmente en horas de la mañana",
                    "Protegerse del frío durante la noche",
                    "Buen momento para turismo con preparación adecuada",
                    "Precaución con el mal de altura en visitantes",
                    "Mantener rutinas normales con protección térmica"
                ]
            },
            (15, 20): {
                'context': "Temperatura agradable para Cusco - Buenas condiciones para turismo",
                'severity': "NORMAL",
                'recommendations': [
                    "Ideal para visitar Machu Picchu y otros atractivos",
                    "Buen momento para caminatas y excursiones",
                    "Condiciones óptimas para fotografía",
                    "Disfrutar de actividades culturales al aire libre",
                    "Aprovechar el clima para explorar la región"
                ]
            },
            (20, 24): {
                'context': "Temperatura cálida inusual para Cusco - Posible evento de calor anómalo",
                'severity': "MODERADO",
                'recommendations': [
                    "Mantenerse hidratado en altitud",
                    "Protegerse del sol fuerte",
                    "Evitar sobre-esfuerzo físico",
                    "Buscar sombra durante el mediodía",
                    "Monitorear síntomas de insolación"
                ]
            },
            (24, 28): {
                'context': "Temperatura muy alta para Cusco - Evento extremadamente raro",
                'severity': "ALTO",
                'recommendations': [
                    "Evitar actividades al mediodía",
                    "Buscar sombra y lugares frescos",
                    "Reducir actividad física intensa",
                    "Protegerse con ropa ligera y sombrero",
                    "Monitorear a personas con condiciones médicas"
                ]
            },
            (28, 32): {
                'context': "Temperatura extrema para Cusco - Evento sin precedentes históricos",
                'severity': "EXTREMO",
                'recommendations': [
                    "Evitar exposición solar directa",
                    "Permanecer en interiores con ventilación",
                    "Consultar autoridades locales sobre recomendaciones",
                    "Proteger especialmente a turistas y visitantes",
                    "Monitorear sistemas de agua por posible evaporación"
                ]
            },
            (32, 36): {
                'context': "Temperatura récord para Cusco - Situación de emergencia",
                'severity': "CRÍTICO",
                'recommendations': [
                    "Activar planes de emergencia por calor extremo",
                    "Evitar cualquier exposición al sol",
                    "Proteger infraestructura turística",
                    "Monitorear constantemente a grupos vulnerables",
                    "Seguir protocolos de emergencia sanitaria"
                ]
            },
            (36, 40): {
                'context': "Temperatura extrema sin precedentes - Peligro vital",
                'severity': "EMERGENCIA",
                'recommendations': [
                    "Declarar emergencia regional por calor extremo",
                    "Evitar actividades exteriores completamente",
                    "Activar sistemas de refugios climáticos",
                    "Monitorear población constantemente",
                    "Preparar para posibles evacuaciones"
                ]
            },
            (40, 45): {
                'context': "Temperatura letal para Cusco - Catástrofe climática",
                'severity': "CATÁSTROFE",
                'recommendations': [
                    "Activar emergencia nacional",
                    "Evacuar zonas de alto riesgo",
                    "Proteger infraestructura crítica",
                    "Monitorear situación en tiempo real",
                    "Solicitar ayuda internacional si es necesario"
                ]
            }
        },
        'arequipa': {
            (0, 8): {
                'context': "Temperatura extremadamente fría para Arequipa - Heladas atípicas",
                'severity': "EXTREMO",
                'recommendations': [
                    "Proteger cultivos de valle con cubiertas térmicas",
                    "Precaución en zonas altas y quebradas",
                    "Evitar exposición nocturna en áreas despejadas",
                    "Proteger sistemas de riego del congelamiento",
                    "Monitorear comunidades rurales y ganado"
                ]
            },
            (8, 14): {
                'context': "Temperatura fría para Arequipa - Posibles heladas en zonas altas",
                'severity': "ALTO",
                'recommendations': [
                    "Usar abrigos en horas de la mañana",
                    "Proteger plantas sensibles en zonas altas",
                    "Conducir con precaución por posible hielo",
                    "Ventilar espacios moderadamente",
                    "Proteger a personas mayores y niños"
                ]
            },
            (14, 18): {
                'context': "Temperatura fresca para Arequipa - Condiciones normales de temporada",
                'severity': "MODERADO",
                'recommendations': [
                    "Buen momento para turismo regional",
                    "Ideal para actividades agrícolas",
                    "Condiciones agradables para caminatas",
                    "Visitar el Cañón del Colca con preparación",
                    "Aprovechar para fotografía de paisajes"
                ]
            },
            (18, 22): {
                'context': "Temperatura agradable para Arequipa - Condiciones ideales",
                'severity': "NORMAL",
                'recommendations': [
                    "Excelente para visitar el Cañón del Colca",
                    "Buen momento para fotografía de paisajes",
                    "Condiciones óptimas para agricultura",
                    "Ideal para actividades al aire libre",
                    "Disfrutar de la gastronomía local en terrazas"
                ]
            },
            (22, 26): {
                'context': "Temperatura cálida para Arequipa - Temporada de verano",
                'severity': "MODERADO",
                'recommendations': [
                    "Mantenerse hidratado",
                    "Protegerse del sol intenso",
                    "Realizar actividades en horas tempranas",
                    "Usar protector solar y sombrero",
                    "Buscar sombra durante el mediodía"
                ]
            },
            (26, 30): {
                'context': "Temperatura muy alta para Arequipa - Ola de calor inusual",
                'severity': "ALTO",
                'recommendations': [
                    "Evitar actividades al mediodía",
                    "Buscar lugares con sombra y ventilación",
                    "Vigilar signos de agotamiento por calor",
                    "Proteger cultivos con sistemas de riego",
                    "Monitorear a grupos vulnerables"
                ]
            },
            (30, 34): {
                'context': "Temperatura extrema para Arequipa - Evento histórico de calor",
                'severity': "EXTREMO",
                'recommendations': [
                    "Permanecer en interiores con aire acondicionado",
                    "Evitar exposición solar directa",
                    "Proteger infraestructura y cultivos",
                    "Seguir recomendaciones de salud",
                    "Monitorear sistemas de agua"
                ]
            },
            (34, 38): {
                'context': "Temperatura récord para Arequipa - Situación crítica",
                'severity': "CRÍTICO",
                'recommendations': [
                    "Activar planes de emergencia por calor",
                    "Evitar cualquier exposición al calor extremo",
                    "Proteger infraestructura crítica",
                    "Monitorear población constantemente",
                    "Seguir protocolos de emergencia"
                ]
            },
            (38, 42): {
                'context': "Temperatura extrema sin precedentes - Peligro vital",
                'severity': "EMERGENCIA",
                'recommendations': [
                    "Declarar emergencia regional",
                    "Evitar actividades exteriores",
                    "Activar sistemas de refugios",
                    "Monitorear en tiempo real",
                    "Preparar para asistencia médica"
                ]
            },
            (42, 47): {
                'context': "Temperatura letal para Arequipa - Catástrofe climática",
                'severity': "CATÁSTROFE",
                'recommendations': [
                    "Activar emergencia nacional",
                    "Evacuar zonas de alto riesgo",
                    "Proteger infraestructura esencial",
                    "Solicitar ayuda internacional",
                    "Monitorear situación continuamente"
                ]
            }
        },
        'iquitos': {
            (0, 16): {
                'context': "Temperatura extremadamente fría para Iquitos - Evento sin precedentes",
                'severity': "EXTREMO",
                'recommendations': [
                    "Protegerse con ropa abrigada y térmica",
                    "Precaución con enfermedades respiratorias",
                    "Evitar baños en ríos y quebradas",
                    "Proteger flora y fauna tropical",
                    "Monitorear comunidades indígenas"
                ]
            },
            (16, 20): {
                'context': "Temperatura fría atípica para Iquitos - Influencia de frentes fríos",
                'severity': "ALTO",
                'recommendations': [
                    "Usar ropa ligera pero con manga larga",
                    "Protegerse de la humedad extrema",
                    "Evitar cambios bruscos de temperatura",
                    "Precaución con enfermedades tropicales",
                    "Proteger cultivos de selva"
                ]
            },
            (20, 24): {
                'context': "Temperatura fresca para Iquitos - Temporada de menor calor",
                'severity': "MODERADO",
                'recommendations': [
                    "Buen momento para ecoturismo",
                    "Ideal para caminatas en selva",
                    "Menor actividad de insectos",
                    "Buen momento para explorar reservas",
                    "Disfrutar de biodiversidad tropical"
                ]
            },
            (24, 28): {
                'context': "Temperatura normal para Iquitos - Condiciones típicas de selva",
                'severity': "NORMAL",
                'recommendations': [
                    "Usar repelente contra mosquitos",
                    "Mantenerse hidratado",
                    "Protegerse del sol tropical",
                    "Disfrutar de actividades en la selva",
                    "Tomar precauciones con fauna silvestre"
                ]
            },
            (28, 32): {
                'context': "Temperatura cálida para Iquitos - Temporada de mayor calor",
                'severity': "MODERADO",
                'recommendations': [
                    "Evitar actividades al mediodía",
                    "Buscar sombra y lugares frescos",
                    "Aumentar ingesta de líquidos",
                    "Protegerse del sol intenso",
                    "Usar ropa ligera y transpirable"
                ]
            },
            (32, 36): {
                'context': "Temperatura muy alta para Iquitos - Ola de calor tropical",
                'severity': "ALTO",
                'recommendations': [
                    "Permanecer en interiores con ventilación",
                    "Evitar actividades físicas intensas",
                    "Protegerse de la insolación",
                    "Monitorear signos de deshidratación",
                    "Cuidado con aumento de insectos"
                ]
            },
            (36, 40): {
                'context': "Temperatura extrema para Iquitos - Evento de calor severo",
                'severity': "EXTREMO",
                'recommendations': [
                    "Evitar cualquier exposición al sol",
                    "Permanecer en lugares con aire acondicionado",
                    "Protegerse de la humedad extrema",
                    "Monitorear sistemas de agua potable",
                    "Seguir recomendaciones de salud tropical"
                ]
            },
            (40, 44): {
                'context': "Temperatura récord para Iquitos - Situación crítica",
                'severity': "CRÍTICO",
                'recommendations': [
                    "Activar planes de emergencia por calor extremo",
                    "Evitar actividades exteriores completamente",
                    "Proteger infraestructura de salud",
                    "Monitorear comunidades ribereñas",
                    "Preparar para posibles inundaciones"
                ]
            },
            (44, 48): {
                'context': "Temperatura extrema sin precedentes - Peligro vital",
                'severity': "EMERGENCIA",
                'recommendations': [
                    "Declarar emergencia regional",
                    "Evacuar zonas de alto riesgo",
                    "Proteger infraestructura crítica",
                    "Monitorear ríos y quebradas",
                    "Preparar para asistencia humanitaria"
                ]
            },
            (48, 52): {
                'context': "Temperatura letal para Iquitos - Catástrofe climática",
                'severity': "CATÁSTROFE",
                'recommendations': [
                    "Activar emergencia nacional",
                    "Evacuar zonas vulnerables",
                    "Proteger ecosistemas frágiles",
                    "Solicitar ayuda internacional",
                    "Monitorear situación en tiempo real"
                ]
            }
        },
        'trujillo': {
            (0, 12): {
                'context': "Temperatura extremadamente fría para Trujillo - Evento muy inusual",
                'severity': "EXTREMO",
                'recommendations': [
                    "Proteger cultivos costeros",
                    "Precaución en transporte marítimo",
                    "Evitar exposición nocturna en playas",
                    "Proteger sistemas de irrigación",
                    "Monitorear comunidades pesqueras"
                ]
            },
            (12, 16): {
                'context': "Temperatura fría para Trujillo - Influencia de corriente de Humboldt",
                'severity': "ALTO",
                'recommendations': [
                    "Usar abrigos ligeros y chaquetas",
                    "Protegerse de la humedad costera",
                    "Conducir con precaución por neblina",
                    "Proteger cultivos sensibles",
                    "Cuidado con actividades marítimas"
                ]
            },
            (16, 20): {
                'context': "Temperatura agradable para Trujillo - Condiciones normales",
                'severity': "MODERADO",
                'recommendations': [
                    "Excelente para visitar playas",
                    "Buen momento para turismo",
                    "Ideal para actividades al aire libre",
                    "Disfrutar de gastronomía costeña",
                    "Aprovechar para deportes acuáticos"
                ]
            },
            (20, 24): {
                'context': "Temperatura cálida para Trujillo - Temporada de verano",
                'severity': "NORMAL",
                'recommendations': [
                    "Protegerse del sol",
                    "Mantenerse hidratado",
                    "Disfrutar de playas con precaución",
                    "Usar protector solar regularmente",
                    "Evitar exposición prolongada al sol"
                ]
            },
            (24, 28): {
                'context': "Temperatura muy alta para Trujillo - Posible ola de calor",
                'severity': "MODERADO",
                'recommendations': [
                    "Evitar actividades al mediodía",
                    "Buscar sombra y lugares frescos",
                    "Vigilar signos de agotamiento",
                    "Protegerse con sombrero y gafas",
                    "Aumentar ingesta de líquidos"
                ]
            },
            (28, 32): {
                'context': "Temperatura extrema para Trujillo - Ola de calor severa",
                'severity': "ALTO",
                'recommendations': [
                    "Permanecer en interiores con aire acondicionado",
                    "Evitar exposición solar directa",
                    "Proteger a grupos vulnerables",
                    "Monitorear sistemas de agua",
                    "Seguir alertas de autoridades"
                ]
            },
            (32, 36): {
                'context': "Temperatura récord para Trujillo - Evento histórico de calor",
                'severity': "EXTREMO",
                'recommendations': [
                    "Activar planes de emergencia por calor",
                    "Evitar cualquier exposición al calor extremo",
                    "Proteger infraestructura costera",
                    "Monitorear playas y zonas turísticas",
                    "Seguir protocolos de emergencia"
                ]
            },
            (36, 40): {
                'context': "Temperatura extrema sin precedentes - Situación crítica",
                'severity': "CRÍTICO",
                'recommendations': [
                    "Declarar emergencia regional",
                    "Evitar actividades exteriores",
                    "Proteger infraestructura esencial",
                    "Monitorear población constantemente",
                    "Preparar para asistencia médica"
                ]
            },
            (40, 44): {
                'context': "Temperatura letal para Trujillo - Peligro vital",
                'severity': "EMERGENCIA",
                'recommendations': [
                    "Activar emergencia nacional",
                    "Evacuar zonas de alto riesgo",
                    "Proteger sistemas de agua y energía",
                    "Monitorear situación en tiempo real",
                    "Solicitar ayuda si es necesario"
                ]
            },
            (44, 48): {
                'context': "Temperatura extrema sin precedentes - Catástrofe climática",
                'severity': "CATÁSTROFE",
                'recommendations': [
                    "Activar emergencia nacional completa",
                    "Evacuar áreas costeras vulnerables",
                    "Proteger infraestructura crítica",
                    "Solicitar ayuda internacional",
                    "Monitorear continuamente la situación"
                ]
            }
        }
    }
    
    # Obtener datos históricos de la región
    region_data = historical_records.get(region.lower(), historical_records['lima'])
    
    # Obtener los contextos específicos para la región
    region_contexts = contexts.get(region.lower(), contexts['lima'])
    
    # Buscar el rango de temperatura correspondiente
    for temp_range, data in region_contexts.items():
        if temp_range[0] <= temperature < temp_range[1]:
            # Formatear la respuesta completa
            response = f"🌡️ {data['context']}\n"
            response += f"⚠️ Nivel de severidad: {data['severity']}\n\n"
            response += f"📊 Datos históricos para {region.capitalize()}:\n"
            response += f"• Temperatura mínima registrada: {region_data['min_temp']}°C\n"
            response += f"• Temperatura máxima registrada: {region_data['max_temp']}°C\n"
            response += f"• Temperatura promedio anual: {region_data['avg_temp']}°C\n\n"
            
            # Añadir eventos extremos relevantes si la temperatura es extrema
            if data['severity'] in ['ALTO', 'EXTREMO', 'CRÍTICO', 'EMERGENCIA', 'CATÁSTROFE']:
                response += f"🚨 Eventos extremos recientes:\n"
                for event in region_data['extreme_events'][-3:]:  # Últimos 3 eventos
                    response += f"• {event['year']}: {event['event']} ({event['temp']}°C)\n"
                response += "\n"
            
            response += f"🛡️ Recomendaciones:\n"
            for i, rec in enumerate(data['recommendations'], 1):
                response += f"{i}. {rec}\n"
            
            # Añadir información adicional según severidad
            if data['severity'] in ['CRÍTICO', 'EMERGENCIA', 'CATÁSTROFE']:
                response += f"\n📞 Contactos de emergencia:\n"
                response += f"• SAMU: 116\n"
                response += f"• Bomberos: 116\n"
                response += f"• Defensa Civil: 115\n"
                response += f"• Policía: 105\n"
            
            return response
    
    # Si la temperatura está fuera de todos los rangos definidos
    if temperature < 0:
        return f"🌡️ Temperatura bajo cero en {region.capitalize()} - Evento extremadamente raro\n\n" + \
               f"⚠️ Nivel de severidad: EXTREMO\n\n" + \
               f"📊 Datos históricos: La temperatura mínima registrada es {region_data['min_temp']}°C\n\n" + \
               f"🛡️ Recomendaciones:\n" + \
               f"1. Evitar cualquier exposición al frío extremo\n" + \
               f"2. Proteger sistemas de agua y cultivos\n" + \
               f"3. Seguir alertas de autoridades\n" + \
               f"4. Monitorear a grupos vulnerables\n" + \
               f"5. Activar planes de emergencia si es necesario"
    elif temperature >= 50:
        return f"🌡️ Temperatura extrema (≥50°C) en {region.capitalize()} - Evento sin precedentes\n\n" + \
               f"⚠️ Nivel de severidad: CATÁSTROFE\n\n" + \
               f"📊 Datos históricos: La temperatura máxima registrada es {region_data['max_temp']}°C\n\n" + \
               f"🛡️ Recomendaciones:\n" + \
               f"1. Activar emergencia nacional inmediatamente\n" + \
               f"2. Evitar cualquier exposición al calor extremo\n" + \
               f"3. Proteger infraestructura crítica\n" + \
               f"4. Monitorear población constantemente\n" + \
               f"5. Solicitar ayuda internacional si es necesario"
    else:
        return f"🌡️ Temperatura dentro del rango regional para {region.capitalize()}\n\n" + \
               f"⚠️ Nivel de severidad: NORMAL\n\n" + \
               f"📊 Rango histórico: {region_data['min_temp']}°C a {region_data['max_temp']}°C\n\n" + \
               f"🛡️ Recomendaciones:\n" + \
               f"1. Mantenerse informado del pronóstico\n" + \
               f"2. Disfrutar de las condiciones climáticas\n" + \
               f"3. Tomar precauciones básicas según la región\n" + \
               f"4. Seguir recomendaciones de autoridades locales"
def get_best_model():
    """Determina el mejor modelo basado en métricas"""
    if not climate_predictor.model_scores:
        return None
    
    best_model = max(
        climate_predictor.model_scores.items(),
        key=lambda x: x[1].get('r2_score', x[1].get('prediction_confidence', 0) / 100)
    )
    
    return {
        'name': best_model[0],
        'metrics': best_model[1]
    }


def calculate_avg_confidence():
    """Calcula confianza promedio de todos los modelos"""
    if not climate_predictor.model_scores:
        return 80.0
    
    confidences = [
        scores.get('prediction_confidence', 80) 
        for scores in climate_predictor.model_scores.values()
    ]
    
    return round(np.mean(confidences), 1)


def calculate_temperature_trend(years, temperatures):
    """Calcula tendencia de temperatura (°C/década)"""
    try:
        if len(years) < 3:
            return 0.0
        
        # Regresión lineal simple
        x = np.array(years) - min(years)
        y = np.array(temperatures)
        
        slope = np.polyfit(x, y, 1)[0]
        trend_per_decade = slope * 10  # Por década
        
        return round(trend_per_decade, 2)
        
    except Exception:
        return 0.0


def calculate_warming_rate(years, temperatures):
    """Calcula tasa de calentamiento por década"""
    trend = calculate_temperature_trend(years, temperatures)
    
    if trend > 0.5:
        return f"+{trend}°C/década (Calentamiento significativo)"
    elif trend > 0.2:
        return f"+{trend}°C/década (Calentamiento moderado)" 
    elif trend > -0.2:
        return f"{trend}°C/década (Estable)"
    else:
        return f"{trend}°C/década (Enfriamiento)"


# Manejo de errores global
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint no encontrado',
        'available_endpoints': [
            '/', '/health', '/models', '/predict', '/predict_map',
            '/historical/<region>', '/download_predictions', '/comparison/<regions>'
        ],
        'status': 'error'
    }), 404
    
@app.route('/debug/models', methods=['GET'])
def debug_models():
    """Verificar estado real de los modelos"""
    status = {}
    
    for name, model in climate_predictor.models.items():
        try:
            # Verificar si es modelo real o fallback
            is_sklearn = hasattr(model, 'estimators_') or hasattr(model, 'coef_')
            
            status[name] = {
                'loaded': True,
                'type': str(type(model).__name__),
                'is_real_model': is_sklearn,
                'has_scaler': name in climate_predictor.scalers,
                'feature_count': len(climate_predictor.feature_columns)
            }
        except Exception as e:
            status[name] = {'error': str(e)}
    
    return jsonify({
        'models': status,
        'model_directory_checked': climate_predictor.model_dir,
        'total_models': len(climate_predictor.models)
    })


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {error}")
    return jsonify({
        'error': 'Error interno del servidor',
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("🌡️ INICIANDO DASHBOARD CLIMÁTICO DEL PERÚ")
    logger.info("="*60)
    logger.info(f"Modelos cargados: {len(climate_predictor.models)}")
    logger.info(f"Regiones históricas: {len(climate_predictor.historical_data)}")
    logger.info(f"Características: {len(climate_predictor.feature_columns)}")
    logger.info("Dashboard disponible en: http://localhost:5000")
    logger.info("API documentación: http://localhost:5000/health")
    logger.info("="*60)
    
    # Configurar y iniciar servidor
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False  # Evitar recargas múltiples
        )
    except Exception as e:
        logger.error(f"Error iniciando servidor: {e}")
        print("Error crítico: No se pudo iniciar el servidor Flask")