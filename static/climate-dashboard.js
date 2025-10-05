// ========================================
// VARIABLES GLOBALES
// ========================================
let predictionChart, temperatureTrendChart, comparisonChart, historicalChart, map;
let currentModel = 'random_forest';
let historicalData = {};
let systemInfo = {};
let mapPredictions = [];

const API_BASE = window.location.origin;

const regionCoords = {
    'peru': { lat: -9.1900, lng: -75.0152, name: 'Perú' },
    'lima': { lat: -12.0464, lng: -77.0428, name: 'Lima' },
    'cusco': { lat: -13.5319, lng: -71.9675, name: 'Cusco' },
    'arequipa': { lat: -16.4090, lng: -71.5375, name: 'Arequipa' },
    'iquitos': { lat: -3.7437, lng: -73.2516, name: 'Iquitos' },
    'trujillo': { lat: -8.1116, lng: -79.0290, name: 'Trujillo' }
};


const dataCache = {
    historical: {},
    lastFetch: {}
};

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function capitalizeFirst(str) {
    if (!str || typeof str !== 'string') return str;
    return str.charAt(0).toUpperCase() + str.slice(1).replace('_', ' ');
}

function getBaseTemperature(region) {
    const baseTemps = {
        'peru': 18.5, 'lima': 19.0, 'cusco': 15.0,
        'arequipa': 18.0, 'iquitos': 26.0, 'trujillo': 22.0
    };
    return baseTemps[region] || 18.5;
}

// ========================================
// INICIALIZACIÓN
// ========================================
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(() => {
        initializeSystem();
        setupEventListeners();
        initializeCharts();
        loadSystemInfo();
        loadHistoricalData();
    }, 500);
});

function initializeSystem() {
    showNotification('Inicializando sistema de predicción climática avanzado...', 'info');
}

function setupEventListeners() {
    // Selector de modelos
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', function() {
            if (this.classList.contains('disabled')) {
                showNotification('Modelo no disponible', 'warning');
                return;
            }
            document.querySelectorAll('.model-card').forEach(c => c.classList.remove('active'));
            this.classList.add('active');
            currentModel = this.dataset.model;
            showMessage(`Modelo ${this.querySelector('.name').textContent} seleccionado`, 'success');
        });
    });

    // Validación de año
    document.getElementById('year').addEventListener('input', function() {
        const year = parseInt(this.value);
        this.style.borderColor = (year >= 2024 && year <= 2100) ? 
            'rgba(78, 205, 196, 0.8)' : 'rgba(255, 107, 107, 0.8)';
    });

    // Cambio automático de datos
    ['region', 'year', 'month'].forEach(id => {
        document.getElementById(id).addEventListener('change', function() {
            loadHistoricalData();
            if (map) updateMap();
        });
    });

    // Tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.dataset.tab;
            
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            this.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            if (targetTab === 'map') {
                setTimeout(() => { initializeMap(); updateMapPredictions(); }, 100);
            } else if (targetTab === 'comparison') {
                loadComparisonData();
            } else if (targetTab === 'historical') {
                updateHistoricalAnalysis();
            }
        });
    });
}

// ========================================
// INICIALIZACIÓN DE GRÁFICOS
// ========================================
function initializeCharts() {
    if (typeof Chart === 'undefined') {
        console.error('Chart.js no cargado');
        setTimeout(() => initializeCharts(), 1000);
        return;
    }

    const chartConfig = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: 'white' } } },
        scales: {
            x: { ticks: { color: 'white' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
            y: { ticks: { color: 'white' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } }
        }
    };

    // Gráfico de predicción
     predictionChart = new Chart(document.getElementById('predictionChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Datos Históricos',
                data: [],
                borderColor: '#4ecdc4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 3,
                spanGaps: false
            }, {
                label: 'Predicción Futura',
                data: [],
                borderColor: '#ff6b6b',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                borderWidth: 3,
                fill: false,
                tension: 0.4,
                pointRadius: 6,
                spanGaps: true,
                borderDash: [5, 5]
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: { display: true, text: 'Histórico + Predicción Integrada', color: 'white', font: { size: 16 } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) label += ': ';
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(1) + '°C';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });

    // Gráfico de tendencias
    temperatureTrendChart = new Chart(document.getElementById('temperatureTrendChart'), {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Temperatura Mensual',
                data: [],
                backgroundColor: 'rgba(78, 205, 196, 0.6)',
                borderColor: 'rgba(78, 205, 196, 1)',
                borderWidth: 1
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: { display: true, text: 'Tendencia Mensual', color: 'white' }
            }
        }
    });

    // Gráfico de comparación
    comparisonChart = new Chart(document.getElementById('comparisonChart'), {
        type: 'radar',
        data: {
            labels: ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'],
            datasets: []
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: { display: true, text: 'Comparación entre Regiones', color: 'white' }
            },
            scales: {
                r: {
                    ticks: { color: 'white' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });

    // Gráfico histórico
    historicalChart = new Chart(document.getElementById('historicalChart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Temperatura Histórica',
                data: [],
                borderColor: '#45b7d1',
                backgroundColor: 'rgba(69, 183, 209, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            ...chartConfig,
            plugins: {
                ...chartConfig.plugins,
                title: { display: true, text: 'Serie Temporal Histórica', color: 'white' }
            }
        }
    });
}

// ========================================
// CARGA DE DATOS DEL SISTEMA
// ========================================
async function loadSystemInfo() {
    const maxRetries = 3;
    let attempt = 0;
    
    while (attempt < maxRetries) {
        try {
            const response = await fetch(`${API_BASE}/models`);
            const data = await response.json();
            
            if (data.status === 'success') {
                systemInfo = data;
                
                // Actualizar UI
                document.getElementById('modelCount').textContent = data.total_models || 0;
                document.getElementById('regionCount').textContent = 
                    data.system_info?.regions_supported?.length || 6;
                
                if (data.model_scores) {
                    const avgScore = Object.values(data.model_scores)
                        .map(score => score.prediction_confidence || 85)
                        .reduce((a, b) => a + b, 0) / Object.values(data.model_scores).length;
                    document.getElementById('avgAccuracy').textContent = Math.round(avgScore) + '%';
                }
                
                // Validar modelo actual
                if (data.available_models && !data.available_models.includes(currentModel)) {
                    currentModel = data.available_models[0];
                    console.log(`Modelo cambiado a: ${currentModel}`);
                }
                
                return; // Éxito, salir
            }
            
            throw new Error('Respuesta sin status success');
            
        } catch (error) {
            attempt++;
            console.error(`Intento ${attempt}/${maxRetries} fallido:`, error);
            
            if (attempt >= maxRetries) {
                // Usar valores por defecto
                console.warn('Usando valores por defecto');
                document.getElementById('modelCount').textContent = '3';
                document.getElementById('avgAccuracy').textContent = '85%';
                document.getElementById('regionCount').textContent = '6';
                systemInfo = {
                    available_models: ['random_forest', 'ridge', 'linear'],
                    total_models: 3
                };
            } else {
                // Esperar antes de reintentar
                await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
            }
        }
    }
}

async function loadHistoricalData() {
    const region = document.getElementById('region').value;
    
    // Verificar cache (5 minutos de validez)
    if (dataCache.historical[region] && 
        Date.now() - dataCache.lastFetch[region] < 300000) {
        updateHistoricalChart(dataCache.historical[region]);
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/historical/${region}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            dataCache.historical[region] = data;
            dataCache.lastFetch[region] = Date.now();
            historicalData[region] = data;
            updateHistoricalChart(data);
            showMessage(`Datos cargados para ${data.region_display_name}`, 'success');
        }
    } catch (error) {
        console.error('Error:', error);
        generateFallbackHistoricalData(region);
    }
}

function generateFallbackHistoricalData(region) {
    const years = [];
    const temperatures = [];
    const baseTemp = getBaseTemperature(region);
    
    for (let year = 1990; year <= 2023; year++) {
        years.push(year);
        const trend = (year - 1990) * 0.03;
        const variation = Math.sin((year - 1990) * 0.5) * 0.8 + (Math.random() - 0.5) * 1.2;
        temperatures.push(parseFloat((baseTemp + trend + variation).toFixed(1)));
    }
    
    historicalData[region] = {
        years,
        temperatures,
        region,
        region_display_name: capitalizeFirst(region),
        statistics: {
            mean_temperature: temperatures.reduce((a, b) => a + b, 0) / temperatures.length,
            warming_rate_per_decade: 0.3
        }
    };
    updateHistoricalChart(historicalData[region]);
}

function getBaseTemperature(region) {
    const baseTemps = {
        'peru': 18.5, 'lima': 19.0, 'cusco': 15.0,
        'arequipa': 18.0, 'iquitos': 26.0, 'trujillo': 22.0
    };
    return baseTemps[region] || 18.5;
}

function updateHistoricalChart(data) {
    historicalChart.data.labels = data.years;
    historicalChart.data.datasets[0].data = data.temperatures;
    historicalChart.update('active');
}

// ========================================
// FUNCIÓN PRINCIPAL DE PREDICCIÓN
// ========================================
async function makePrediction() {
    const year = parseInt(document.getElementById('year').value);
    const region = document.getElementById('region').value;
    const month = parseInt(document.getElementById('month').value);
    
    // Validación mejorada
    if (!year || isNaN(year) || year < 2024 || year > 2100) {
        showNotification('Año invalido. Debe estar entre 2024 y 2100', 'error');
        document.getElementById('year').focus();
        document.getElementById('year').style.borderColor = '#ef4444';
        return;
    }
    
    if (!region) {
        showNotification('Selecciona una region', 'error');
        return;
    }
    
    if (!month || isNaN(month) || month < 1 || month > 12) {
        showNotification('Mes invalido', 'error');
        return;
    }
    
    // Verificar que tengamos modelos disponibles
    if (!systemInfo.available_models || systemInfo.available_models.length === 0) {
        showNotification('No hay modelos disponibles. Espera un momento...', 'warning');
        await loadSystemInfo();
        
        if (!systemInfo.available_models || systemInfo.available_models.length === 0) {
            showNotification('Error: Sistema sin modelos', 'error');
            return;
        }
    }
    
    // Validar que el modelo seleccionado existe
    if (!systemInfo.available_models.includes(currentModel)) {
        console.warn(`Modelo ${currentModel} no disponible, cambiando a ${systemInfo.available_models[0]}`);
        currentModel = systemInfo.available_models[0];
        
        // Actualizar UI
        document.querySelectorAll('.model-card').forEach(card => {
            card.classList.remove('active');
            if (card.dataset.model === currentModel) {
                card.classList.add('active');
            }
        });
    }
    
    showLoading(true);
    console.log('Iniciando prediccion:', { year, region, month, model: currentModel });
    
    try {
        // Crear features
        const features = createFeatureVector(year, month, region);
        console.log('Features creadas:', features);
        
        // Preparar request
        const requestBody = {
            features: features,
            model: currentModel,
            region: region
        };
        
        // Hacer request
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log('Response status:', response.status);
        
        // Manejar errores HTTP
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                error: `Error HTTP ${response.status}`
            }));
            
            throw new Error(errorData.error || `Error ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Datos recibidos:', data);
        
        // Validar respuesta
        if (data.status !== 'success' || data.predicted_temperature === undefined) {
            throw new Error('Respuesta invalida del servidor');
        }
        
        // Mostrar resultados
        displayPredictionResult(data);
        updatePredictionChart(year, data.predicted_temperature, region);
        
        // Actualizar recomendaciones
        const temp = parseFloat(data.predicted_temperature);
        const alertLevel = data.alert_level || getAlertLevel(temp, region);
        
        displayHealthRecommendations(temp, alertLevel, region, month);
        displayAgricultureRecommendations(temp, region, month);
        updateAirQualityPanel(temp, region, month);
        
        // Mostrar paneles
        const airQualityPanel = document.getElementById('airQualityPanel');
        if (airQualityPanel) airQualityPanel.style.display = 'block';
        
        showNotification(`Prediccion completada: ${temp.toFixed(1)}C`, 'success');
        
    } catch (error) {
        console.error('Error en prediccion:', error);
        showNotification(`Error: ${error.message}`, 'error');
        
        // Fallback
        const fallback = calculateFallbackPrediction(year, region, month);
        const fallbackData = {
            predicted_temperature: fallback.temp,
            confidence: fallback.confidence,
            model_used: 'respaldo local',
            region: region,
            alert_level: getAlertLevel(fallback.temp, region),
            prediction_month: month,
            prediction_date: year,
            climate_context: getClimateContext(fallback.temp, region)
        };
        
        displayPredictionResult(fallbackData);
        updatePredictionChart(year, fallback.temp, region);
        displayHealthRecommendations(fallback.temp, fallbackData.alert_level, region, month);
        displayAgricultureRecommendations(fallback.temp, region, month);
        updateAirQualityPanel(fallback.temp, region, month);
        
        const airQualityPanel = document.getElementById('airQualityPanel');
        if (airQualityPanel) airQualityPanel.style.display = 'block';
        
    } finally {
        showLoading(false);
    }
}


function createFeatureVector(year, month, region) {
    const coords = regionCoords[region] || regionCoords.peru;
    const yearsSince1990 = year - 1990;
    
    return {
        year: year,
        year_since_1990: yearsSince1990,
        decade: Math.floor(yearsSince1990 / 10),
        month: month,
        season: Math.floor((month - 1) / 3) + 1,
        latitude: coords.lat,
        longitude: coords.lng,
        altitude: getRegionAltitude(region),
        altitude_norm: (getRegionAltitude(region) - 1500) / 1500,
        coastal: ['lima', 'trujillo'].includes(region) ? 1 : 0,
        highland: ['cusco', 'arequipa'].includes(region) ? 1 : 0,
        jungle: region === 'iquitos' ? 1 : 0,
        el_nino_index: Math.sin(2 * Math.PI * yearsSince1990 / 4.5),
        solar_cycle: Math.sin(2 * Math.PI * yearsSince1990 / 11),
        linear_trend: yearsSince1990,
        quadratic_trend: yearsSince1990 * yearsSince1990,
        region: region  
    };
}

function getRegionAltitude(region) {
    const altitudes = {
        'lima': 154, 'cusco': 3399, 'arequipa': 2335,
        'iquitos': 106, 'trujillo': 34, 'peru': 500
    };
    return altitudes[region] || 500;
}

function calculateFallbackPrediction(year, region, month) {
    const baseTemp = getBaseTemperature(region);
    const yearsSince1990 = year - 1990;
    const warmingTrend = yearsSince1990 * 0.03;
    const seasonalVariation = Math.sin((month - 1) * Math.PI / 6) * 1.5;
    
    let temp = baseTemp + warmingTrend + seasonalVariation;
    temp = Math.max(8, Math.min(35, temp));
    
    return { temp: parseFloat(temp.toFixed(1)), confidence: 75 };
}

// ========================================
// DISPLAY DE RESULTADOS
// ========================================
function displayPredictionResult(data) {
    const resultDiv = document.getElementById('result');
    const tempValue = document.getElementById('tempValue');
    const tempContext = document.getElementById('tempContext');
    const confidenceBar = document.getElementById('confidence');
    const confidenceText = document.getElementById('confidenceText');
    const modelUsed = document.getElementById('modelUsed');
    const alertIndicator = document.getElementById('alertIndicator');
    
    const temp = parseFloat(data.predicted_temperature);
    
    animateTemperature(tempValue, temp);
    
    const context = data.climate_context || getClimateContext(temp, data.region);
    let alertClass = '';
    let alertText = '';
    
    if (data.alert_level) {
        const alertMap = {
            'extreme_high': { class: 'danger', text: 'CALOR EXTREMO', color: '#f44336' },
            'high': { class: 'warning', text: 'TEMPERATURA ALTA', color: '#ff9800' },
            'extreme_low': { class: 'danger', text: 'FRÍO EXTREMO', color: '#3f51b5' },
            'low': { class: 'warning', text: 'TEMPERATURA BAJA', color: '#2196f3' },
            'normal': { class: '', text: 'NORMAL', color: '#4ecdc4' }
        };
        
        const config = alertMap[data.alert_level] || alertMap.normal;
        alertClass = config.class;
        alertText = config.text;
        tempValue.style.color = config.color;
    }
    
    tempContext.textContent = context;
    alertIndicator.innerHTML = `<span class="alert-indicator alert-${data.alert_level || 'normal'}">${alertText}</span>`;
    
    confidenceBar.style.width = data.confidence + '%';
    confidenceText.textContent = `${data.confidence}% Confianza`;
    
    const monthNames = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];
    const monthName = monthNames[data.prediction_month - 1] || monthNames[5];
    
    modelUsed.textContent = `Modelo: ${data.model_used} | Región: ${data.region} | Fecha: ${monthName} ${data.prediction_date}`;
    
    resultDiv.className = `prediction-result show ${alertClass}`;
}

function getClimateContext(temperature, region) {
    if (temperature > 30) return "Temperatura excepcionalmente alta - Posible ola de calor";
    if (temperature < 10) return "Temperatura muy baja - Condiciones frías extremas";
    if (temperature > 25) return "Temperatura cálida - Por encima del promedio";
    if (temperature < 15) return "Temperatura fresca - Por debajo del promedio";
    return "Temperatura dentro del rango esperado";
}

function getAlertLevel(temp, region) {
    if (temp > 30) return 'extreme_high';
    if (temp > 27) return 'high';
    if (temp < 8) return 'extreme_low';
    if (temp < 12) return 'low';
    return 'normal';
}

function animateTemperature(element, targetTemp) {
    const currentTemp = parseFloat(element.textContent) || 0;
    const duration = 2000;
    const steps = 60;
    const increment = (targetTemp - currentTemp) / steps;
    let step = 0;
    
    const animation = setInterval(() => {
        step++;
        element.textContent = (currentTemp + increment * step).toFixed(1) + '°C';
        if (step >= steps) {
            clearInterval(animation);
            element.textContent = targetTemp.toFixed(1) + '°C';
        }
    }, duration / steps);
}

// ========================================
// NUEVA FUNCIÓN: RECOMENDACIONES DE SALUD
// ========================================
function displayHealthRecommendations(temperature, alertLevel, region, month) {
    // Determinar el tipo de clima basado en la temperatura
    let climateType = 'normal';
    if (temperature < 10) climateType = 'extreme_cold';
    else if (temperature < 15) climateType = 'cold';
    else if (temperature > 28) climateType = 'extreme_heat';
    else if (temperature > 25) climateType = 'heat';
    
    // Recomendaciones específicas según el tipo de clima
    const recommendations = {
        extreme_cold: {
            icon: '❄️',
            title: 'ALERTA DE FRÍO EXTREMO',
            health: [
                'Evitar exposición prolongada al frío, especialmente en horas de la madrugada y noche',
                'Abrigarse adecuadamente con varias capas de ropa (usar lana, térmicas, gorro, guantes y bufanda)',
                'Proteger extremidades (manos, pies, orejas) que son más susceptibles al congelamiento',
                'Mantener la vivienda caliente (18-21°C) usando calefacción segura',
                'Consumir bebidas calientes y alimentos energéticos (sopas, caldos, infusiones)',
                'Estar atento a síntomas de hipotermia (temblores, confusión, piel fría y pálida)',
                'Verificar el estado de personas mayores y niños que son más vulnerables'
            ],
            livestock: [
                'Proteger cultivos con cubiertas térmicas o invernaderos temporales',
                'Asegurar refugio cálido para el ganado, con paja o materiales aislantes',
                'Incrementar alimentación energética para animales (forrajes de mayor calidad, suplementos)',
                'Proteger fuentes de agua del congelamiento (usar calentadores o cubrir)',
                'Vigilar signos de estrés por frío en animales (temblores, agrupamiento excesivo)'
            ],
            infrastructure: [
                'Proteger tuberías para evitar congelamiento (aislar con espuma o trapos)',
                'Verificar sistemas de calefacción y tener a mano generadores de energía',
                'Aislar viviendas y edificios con burbujas, plásticos o materiales aislantes',
                'Tener a mano mantas, alimentos no perecederos y agua potable',
                'Mantener los desagües libres para evitar acumulación de hielo'
            ],
            color: '#3f51b5'
        },
        cold: {
            icon: '🌡️',
            title: 'Temperatura Baja',
            health: [
                'Abrigarse adecuadamente con ropa de abrigo',
                'Evitar cambios bruscos de temperatura al entrar o salir de lugares cerrados',
                'Mantener ambientes interiores calientes (mínimo 16°C)',
                'Proteger vías respiratorias con bufandas o mascarillas en ambientes fríos',
                'Consumir alimentos calientes y bebidas tibias para mantener la temperatura corporal',
                'Evitar el consumo excesivo de alcohol ya que dilata los vasos sanguíneos y aumenta la pérdida de calor'
            ],
            livestock: [
                'Proteger cultivos sensibles al frío con cubiertas plásticas',
                'Proporcionar refugio adicional al ganado durante la noche',
                'Aumentar ligeramente la ración de alimento para compensar el gasto energético',
                'Verificar que los sistemas de abastecimiento de agua no se congelen',
                'Monitorear el estado de salud de los animales diariamente'
            ],
            infrastructure: [
                'Revisar el aislamiento térmico de edificios y viviendas',
                'Realizar mantenimiento preventivo de sistemas de calefacción',
                'Proteger tuberías expuestas con materiales aislantes',
                'Tener a mano materiales de emergencia para reparaciones rápidas',
                'Verificar el estado de techos y estructuras que puedan ser afectadas por vientos fríos'
            ],
            color: '#2196f3'
        },
        normal: {
            icon: '✅',
            title: 'Temperatura Normal',
            health: [
                'Condiciones óptimas para actividades al aire libre',
                'Mantener una buena hidratación durante todo el día',
                'Aprovechar para realizar ejercicio físico y actividades recreativas',
                'Condiciones favorables para la salud general y el bienestar',
                'Mantener una dieta equilibrada y hábitos de vida saludables',
                'Realizar actividades de prevención de enfermedades de forma regular'
            ],
            livestock: [
                'Condiciones ideales para la mayoría de cultivos',
                'Mantener riego normal según cronograma establecido',
                'Ganado en condiciones óptimas de salud y producción',
                'Buen momento para realizar siembras y cosechas',
                'Realizar labores de mantenimiento de infraestructura agrícola',
                'Aprovechar para realizar mejoras en sistemas de riego'
            ],
            infrastructure: [
                'Operación normal de infraestructuras y servicios públicos',
                'Buen momento para realizar mantenimiento preventivo de edificios',
                'Condiciones favorables para trabajos de construcción y reparación',
                'Sistemas de energía y agua funcionando en condiciones óptimas',
                'Oportunidad para realizar mejoras en infraestructura vial',
                'Buen momento para inspecciones técnicas de estructuras'
            ],
            color: '#4ecdc4'
        },
        heat: {
            icon: '☀️',
            title: 'Temperatura Alta',
            health: [
                'Mantenerse hidratado constantemente (beber 2-3 litros de agua diariamente)',
                'Evitar exposición solar directa entre las 11am y 4pm',
                'Usar protector solar FPS 50+ y reaplicar cada 2 horas',
                'Vestir ropa ligera, de colores claros y tejidos transpirables',
                'Buscar lugares con sombra o aire acondicionado durante las horas más calurosas',
                'Reducir la actividad física intensa durante las horas pico de calor',
                'Estar atento a síntomas de agotamiento por calor (mareos, náuseas, fatiga extrema)'
            ],
            livestock: [
                'Aumentar la frecuencia de riego para compensar la evaporación',
                'Proporcionar sombra adecuada al ganado (árboles, techados, sombrillas)',
                'Realizar trabajos agrícolas en horas tempranas de la mañana o al atardecer',
                'Monitorear signos de estrés hídrico en plantas (hojas marchitas, crecimiento lento)',
                'Asegurar abundante agua fresca para los animales y renovarla varias veces al día',
                'Evitar el transporte de animales durante las horas más calurosas'
            ],
            infrastructure: [
                'Verificar sistemas de refrigeración y aire acondicionado',
                'Proteger equipos electrónicos del sobrecalentamiento',
                'Realizar mantenimiento de sistemas de ventilación industrial',
                'Verificar redes eléctricas por posible aumento de demanda',
                'Proteger materiales sensibles al calor en almacenes y obras',
                'Asegurar el funcionamiento adecuado de sistemas de enfriamiento de maquinaria'
            ],
            color: '#ff9800'
        },
        extreme_heat: {
            icon: '🔥',
            title: 'ALERTA DE CALOR EXTREMO',
            health: [
                'EVITAR exposición al sol entre las 10am y 5pm',
                'Permanecer en lugares con aire acondicionado o buena ventilación',
                'Beber 2-3 litros de agua diariamente, incluso sin sentir sed',
                'Usar ropa fresca, ligera y de colores claros (algodón, lino)',
                'Vigilar signos de golpe de calor (fiebre alta, confusión, pérdida de conciencia)',
                'Proteger a grupos vulnerables (niños, ancianos, personas con enfermedades crónicas)',
                'Tomar duchas o baños frescos para bajar la temperatura corporal',
                'Evitar bebidas alcohólicas, cafeína o muy azucaradas'
            ],
            livestock: [
                'Realizar riego nocturno o en las primeras horas de la mañana',
                'Proporcionar sombra obligatoria para todo el ganado',
                'Suspender trabajos agrícolas intensivos durante las horas más calurosas',
                'Monitoreo constante de estrés térmico en animales (respiración acelerada, letargo)',
                'Preparar para posibles incendios forestales (limpiar maleza, tener extintores)',
                'Aumentar la frecuencia de cambio de agua para garantizar que esté fresca',
                'Considerar la reducción de actividades que expongan a los animales al calor'
            ],
            infrastructure: [
                'Preparar sistemas de enfriamiento de emergencia para centros críticos',
                'Verificar la capacidad de las redes eléctricas ante el aumento de demanda',
                'Tener planes de contingencia por posibles cortes de energía',
                'Proteger materiales sensibles al calor con coberturas térmicas',
                'Monitorear el estado de puentes y estructuras metálicas que se expanden con el calor',
                'Asegurar el funcionamiento de sistemas de protección contra incendios',
                'Realizar inspecciones de equipos que puedan sobrecalentarse'
            ],
            color: '#f44336'
        }
    };

    // Recomendaciones específicas por región
    let regionalNote = '';
    if (region === 'cusco' || region === 'arequipa') {
        regionalNote = '<p><strong>📍 Sierra:</strong> Mayor amplitud térmica día/noche. Prepararse para frío nocturno incluso con días cálidos. La altitud aumenta el riesgo de insolación y deshidratación. Consumir mate de coca para adaptarse a la altura.</p>';
    } else if (region === 'iquitos') {
        regionalNote = '<p><strong>📍 Selva:</strong> Alta humedad. Aumentar hidratación y vigilar enfermedades tropicales (dengue, malaria). Usar repelente y mosquiteros. Evitar acumulación de agua estancada para prevenir criaderos de mosquitos.</p>';
    } else if (region === 'lima' || region === 'trujillo') {
        regionalNote = '<p><strong>📍 Costa:</strong> Influencia de corriente de Humboldt. Neblina matinal posible en temporada de invierno (mayo-agosto). La humedad relativa puede aumentar la sensación térmica. Precaución con la radiación UV que es alta durante todo el año.</p>';
    }

    // Recomendaciones por mes
    let seasonalNote = '';
    const monthNames = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];
    
    if (month >= 12 || month <= 3) {
        seasonalNote = `<p><strong>📅 ${monthNames[month-1]}:</strong> Temporada de verano. Mayor radiación solar y posibles precipitaciones en sierra y selva. Precaución con lluvias intensas en la sierra y aumento de enfermedades transmitidas por mosquitos en la selva.</p>`;
    } else if (month >= 6 && month <= 8) {
        seasonalNote = `<p><strong>📅 ${monthNames[month-1]}:</strong> Temporada de invierno. Posibles heladas en sierra y neblina en costa. Aumento de enfermedades respiratorias. Precaución con el frío en zonas de altitud.</p>`;
    }

    // Recomendaciones basadas en datos históricos de Perú
    let historicalNote = '';
    const historicalData = getHistoricalDataForRegion(region);
    if (historicalData) {
        const avgTemp = historicalData.avgTemperature;
        const trend = historicalData.trend;
        const extremeEvents = historicalData.extremeEvents || [];
        
        historicalNote = `
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                <h4 style="color: #ff6b6b; margin-top: 0;">📊 Datos Históricos para ${capitalizeFirst(region)}</h4>
                <p><strong>Temperatura promedio histórica:</strong> ${avgTemp}°C</p>
                <p><strong>Tendencia de calentamiento:</strong> ${trend > 0 ? '+' : ''}${trend}°C por década</p>
                <p><strong>Eventos extremos registrados:</strong> ${extremeEvents.length} en los últimos 30 años</p>
                ${extremeEvents.length > 0 ? `<p><strong>Principales eventos:</strong> ${extremeEvents.slice(0, 3).join(', ')}</p>` : ''}
            </div>
        `;
    }

    const rec = recommendations[climateType];
    
    const html = `
        <div class="recommendations-card" style="background: linear-gradient(135deg, ${rec.color}15, ${rec.color}05); border-left: 4px solid ${rec.color}; padding: 25px; border-radius: 12px; margin-bottom: 20px;">
            <h2 style="color: ${rec.color}; margin: 0 0 15px 0;">${rec.icon} ${rec.title}</h2>
            
            ${historicalNote}
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                ${regionalNote}
                ${seasonalNote}
                <p><strong>ℹ️ Información:</strong> Temperatura predicha ${temperature.toFixed(1)}°C en ${capitalizeFirst(region)}</p>
            </div>
            
            <div style="margin-top: 20px;">
                <h3 style="color: #4ecdc4; margin-bottom: 12px;">🏥 Recomendaciones de Salud Pública:</h3>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                    ${rec.health.map(item => `<li style="margin: 10px 0;">${item}</li>`).join('')}
                </ul>
            </div>
            
            <div style="margin-top: 20px;">
                <h3 style="color: #96ceb4; margin-bottom: 12px;">🐄 Impacto Agrícola y Ganadero:</h3>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                    ${rec.livestock.map(item => `<li style="margin: 10px 0;">${item}</li>`).join('')}
                </ul>
            </div>
            
            <div style="margin-top: 20px;">
                <h3 style="color: #ffc107; margin-bottom: 12px;">🏗️ Infraestructura y Servicios:</h3>
                <ul style="margin: 0; padding-left: 20px; line-height: 1.8;">
                    ${rec.infrastructure.map(item => `<li style="margin: 10px 0;">${item}</li>`).join('')}
                </ul>
            </div>
            
            <div style="margin-top: 25px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <h4 style="color: #ff6b6b; margin-top: 0;">🚨 Nivel de Alerta: ${alertLevel.replace('_', ' ').toUpperCase()}</h4>
                <p>Las recomendaciones anteriores están basadas en las condiciones climáticas pronosticadas. 
                Monitoree las actualizaciones del servicio meteorológico y siga las indicaciones de las autoridades locales.</p>
                
                <div style="margin-top: 15px;">
                    <h5 style="color: #ffc107;">📞 Contactos de Emergencia:</h5>
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>SAMU (Emergencias médicas):</strong> 116</li>
                        <li><strong>Bomberos:</strong> 116</li>
                        <li><strong>Policía Nacional:</strong> 105</li>
                        <li><strong>Defensa Civil:</strong> 115</li>
                        <li><strong>INDECI (Instituto Nacional de Defensa Civil):</strong> 595-0000</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    document.getElementById('healthRecommendations').innerHTML = html;
}

// Función auxiliar para obtener datos históricos por región
function getHistoricalDataForRegion(region) {

    const historicalData = {
        lima: {
            avgTemperature: 19.0,
            trend: 0.15,
            extremeEvents: ['Ola de calor 2017', 'Lluvias intensas 2017', 'Helada atípica 2018']
        },
        cusco: {
            avgTemperature: 15.0,
            trend: 0.25,
            extremeEvents: ['Heladas 2003', 'Heladas 2013', 'Sequía 2016']
        },
        arequipa: {
            avgTemperature: 18.0,
            trend: 0.18,
            extremeEvents: ['Heladas 2007', 'Sequía 2015', 'Lluvias intensas 2019']
        },
        iquitos: {
            avgTemperature: 26.0,
            trend: 0.20,
            extremeEvents: ['Inundaciones 2012', 'Inundaciones 2015', 'Sequía 2016']
        },
        trujillo: {
            avgTemperature: 22.0,
            trend: 0.17,
            extremeEvents: ['Ola de calor 2016', 'Lluvias intensas 2017', 'Sequía 2014']
        },
        peru: {
            avgTemperature: 20.0,
            trend: 0.19,
            extremeEvents: ['Fenómeno El Niño 1997-98', 'Fenómeno El Niño 2015-16', 'Heladas 2013']
        }
    };
    
    return historicalData[region] || historicalData.peru;
}

// Función auxiliar para capitalizar primera letra
function capitalizeFirst(str) {
    if (!str || typeof str !== 'string') return str;
    return str.charAt(0).toUpperCase() + str.slice(1).replace('_', ' ');
}

// ========================================
// NUEVA FUNCIÓN: CALENDARIO AGRÍCOLA
// ========================================
function displayAgricultureRecommendations(temperature, region, month) {
    const crops = {
        lima: {
            optimal: ['Tomate', 'Ají', 'Espárrago', 'Palta', 'Maíz', 'Frijol'],
            season: month >= 9 && month <= 3 ? 'Temporada de siembra' : 'Temporada de cosecha',
            challenges: ['Suelos salinos', 'Escasez de agua', 'Plagas urbanas'],
            adaptations: ['Riego por goteo', 'Cultivos protegidos', 'Control biológico de plagas'],
            waterNeeds: 'Alta (5-7 mm/día)',
            soilType: 'Arenoso-arcilloso',
            pests: ['Mosca blanca', 'Pulgones', 'Araña roja']
        },
        cusco: {
            optimal: ['Papa', 'Quinua', 'Maíz', 'Habas', 'Cebada', 'Olluco'],
            season: month >= 10 || month <= 3 ? 'Temporada de lluvias - siembra' : 'Temporada seca',
            challenges: ['Heladas', 'Erosión de suelos', 'Plagas andinas'],
            adaptations: ['Cultivos en terrazas', 'Sistemas de riego tecnificado', 'Variedades resistentes al frío'],
            waterNeeds: 'Moderada (3-5 mm/día)',
            soilType: 'Franco-arcilloso',
            pests: ['Gorgojo de los andes', 'Polilla de la papa', 'Barrenador del tallo']
        },
        arequipa: {
            optimal: ['Cebolla', 'Ajo', 'Papa', 'Alfalfa', 'Orégano', 'Maíz'],
            season: month >= 4 && month <= 8 ? 'Cosecha principal' : 'Preparación de suelos',
            challenges: ['Sequía', 'Suelos pedregosos', 'Radiación solar intensa'],
            adaptations: ['Riego por aspersión', 'Cultivos de secano', 'Sombreado parcial'],
            waterNeeds: 'Moderada-alta (4-6 mm/día)',
            soilType: 'Arenoso',
            pests: ['Trips', 'Mosca minadora', 'Nematodos']
        },
        iquitos: {
            optimal: ['Plátano', 'Yuca', 'Cacao', 'Café', 'Pijuayo', 'Cultivos amazónicos'],
            season: 'Producción continua - clima tropical',
            challenges: ['Exceso de humedad', 'Enfermedades fúngicas', 'Plagas tropicales'],
            adaptations: ['Drenaje adecuado', 'Control fitosanitario', 'Sistemas agroforestales'],
            waterNeeds: 'Baja (2-4 mm/día)',
            soilType: 'Arcilloso con alto contenido de materia orgánica',
            pests: ['Sigatoka negra', 'Moniliasis', 'Broca del café']
        },
        trujillo: {
            optimal: ['Arroz', 'Caña de azúcar', 'Mango', 'Palta', 'Algodón', 'Maíz'],
            season: month >= 11 || month <= 4 ? 'Temporada de lluvias' : 'Temporada seca',
            challenges: ['Salinidad', 'Deficiencia de nutrientes', 'Plagas de cultivos'],
            adaptations: ['Rotación de cultivos', 'Enmiendas orgánicas', 'Manejo integrado de plagas'],
            waterNeeds: 'Alta (6-8 mm/día)',
            soilType: 'Franco-arenoso',
            pests: ['Chinche de la caña', 'Picudo del algodonero', 'Mosca de la fruta']
        },
        peru: {
            optimal: ['Papa', 'Maíz', 'Arroz', 'Quinua', 'Café', 'Cacao'],
            season: 'Varía por región',
            challenges: ['Variabilidad climática', 'Fragmentación de tierras', 'Falta de tecnología'],
            adaptations: ['Diversificación productiva', 'Asistencia técnica', 'Acceso a créditos'],
            waterNeeds: 'Variable según región',
            soilType: 'Diverso según región',
            pests: ['Variable según región y cultivo']
        }
    };

    const regionCrops = crops[region] || crops.peru;
    
    let tempImpact = '';
    if (temperature > 28) {
        tempImpact = `
            <div class="alert-box warning">
                <strong>⚠️ Alerta:</strong> Temperatura alta puede causar estrés hídrico. 
                Aumentar riego y considerar cultivos resistentes al calor.
                <br><strong>Recomendación:</strong> Implementar sistemas de riego por goteo y usar mulch para conservar humedad.
            </div>
        `;
    } else if (temperature < 10) {
        tempImpact = `
            <div class="alert-box danger">
                <strong>❄️ Alerta de Helada:</strong> Proteger cultivos sensibles. 
                Considerar variedades resistentes al frío.
                <br><strong>Recomendación:</strong> Usar cubiertas plásticas o túneles bajos para proteger los cultivos.
            </div>
        `;
    } else {
        tempImpact = `
            <div class="alert-box success">
                <strong>✅ Condiciones Favorables:</strong> Temperatura óptima para la mayoría de cultivos.
                <br><strong>Recomendación:</strong> Aprovechar para realizar siembras y labores culturales.
            </div>
        `;
    }

    // Recomendaciones específicas basadas en el mes
    let monthlyAdvice = '';
    const monthNames = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];
    
    if (month === 1 || month === 2) {
        monthlyAdvice = `
            <div class="monthly-advice">
                <h4>📅 Recomendaciones para ${monthNames[month-1]}</h4>
                <p>Temporada de lluvias en sierra y selva. Prevenir enfermedades fúngicas. 
                En la costa, es temporada de cosecha de maíz y frijol.</p>
                <ul>
                    <li>Aplicar fungicidas preventivos en zonas de alta humedad</li>
                    <li>Realizar podas de formación en frutales</li>
                    <li>Preparar suelos para siembras de temporada seca</li>
                </ul>
            </div>
        `;
    } else if (month >= 3 && month <= 5) {
        monthlyAdvice = `
            <div class="monthly-advice">
                <h4>📅 Recomendaciones para ${monthNames[month-1]}</h4>
                <p>Transición a temporada seca. Buen momento para siembras de ciclo corto.</p>
                <ul>
                    <li>Siembra de hortalizas de hoja (lechuga, espinaca)</li>
                    <li>Fertilización de frutales post-cosecha</li>
                    <li>Control de malezas antes de la floración</li>
                </ul>
            </div>
        `;
    } else if (month >= 6 && month <= 8) {
        monthlyAdvice = `
            <div class="monthly-advice">
                <h4>📅 Recomendaciones para ${monthNames[month-1]}</h4>
                <p>Temporada seca en costa y sierra. Cuidado con heladas en zonas altas.</p>
                <ul>
                    <li>Proteger cultivos con cubiertas antiheladas</li>
                    <li>Aumentar frecuencia de riego en zonas costeras</li>
                    <li>Realizar mantenimiento de sistemas de riego</li>
                </ul>
            </div>
        `;
    } else if (month >= 9 && month <= 11) {
        monthlyAdvice = `
            <div class="monthly-advice">
                <h4>📅 Recomendaciones para ${monthNames[month-1]}</h4>
                <p>Inicio de temporada de lluvias. Preparar suelos y planificar siembras.</p>
                <ul>
                    <li>Incorporar materia orgánica a los suelos</li>
                    <li>Siembra de cultivos de temporada</li>
                    <li>Control preventivo de plagas y enfermedades</li>
                </ul>
            </div>
        `;
    }

    const html = `
        <div class="agriculture-panel">
            <h2 style="color: #96ceb4; margin-bottom: 20px;">🌾 Calendario Agrícola - ${capitalizeFirst(region)}</h2>
            
            ${tempImpact}
            ${monthlyAdvice}
            
            <div class="crop-recommendations" style="margin-top: 25px;">
                <h3 style="color: #4ecdc4;">Cultivos Recomendados:</h3>
                <div class="crop-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    ${regionCrops.optimal.map(crop => `
                        <div class="crop-card" style="background: rgba(150, 206, 180, 0.1); padding: 20px; border-radius: 10px; border: 2px solid rgba(150, 206, 180, 0.3);">
                            <div style="font-size: 2em; text-align: center; margin-bottom: 10px;">🌱</div>
                            <div style="font-weight: bold; text-align: center; color: #96ceb4;">${crop}</div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="season-info" style="margin-top: 25px; padding: 20px; background: rgba(78, 205, 196, 0.1); border-radius: 10px;">
                <h3 style="color: #4ecdc4;">📅 Temporada Actual:</h3>
                <p style="font-size: 1.1em; margin-top: 10px;">${regionCrops.season}</p>
                <div style="margin-top: 15px;">
                    <p><strong>Necesidades de agua:</strong> ${regionCrops.waterNeeds}</p>
                    <p><strong>Tipo de suelo:</strong> ${regionCrops.soilType}</p>
                    <p><strong>Principales plagas:</strong> ${regionCrops.pests.join(', ')}</p>
                </div>
            </div>
            
            <div class="challenges-adaptations" style="margin-top: 25px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="padding: 20px; background: rgba(255, 107, 107, 0.1); border-radius: 10px;">
                    <h3 style="color: #ff6b6b;">🌋 Desafíos Principales:</h3>
                    <ul style="line-height: 1.8;">
                        ${regionCrops.challenges.map(challenge => `<li>${challenge}</li>`).join('')}
                    </ul>
                </div>
                <div style="padding: 20px; background: rgba(78, 205, 196, 0.1); border-radius: 10px;">
                    <h3 style="color: #4ecdc4;">🛡️ Adaptaciones Recomendadas:</h3>
                    <ul style="line-height: 1.8;">
                        ${regionCrops.adaptations.map(adaptation => `<li>${adaptation}</li>`).join('')}
                    </ul>
                </div>
            </div>
            
            <div class="irrigation-tips" style="margin-top: 25px; padding: 20px; background: rgba(69, 183, 209, 0.1); border-radius: 10px;">
                <h3 style="color: #45b7d1;">💧 Recomendaciones de Riego:</h3>
                ${getIrrigationAdvice(temperature, region)}
            </div>
        </div>
    `;
    
    document.getElementById('agricultureRecommendations').innerHTML = html;
}

function getIrrigationAdvice(temp, region) {
    if (temp > 28) {
        return `
            <ul style="line-height: 1.8;">
                <li>Aumentar frecuencia de riego en 30-40%</li>
                <li>Regar temprano en la mañana (5-7am) o al atardecer (6-8pm)</li>
                <li>Aplicar mulch (cobertura orgánica) para conservar humedad</li>
                <li>Monitorear signos de estrés hídrico en plantas</li>
                <li>Considerar sistemas de riego automatizados</li>
                <li>Evitar riego durante las horas de máxima radiación solar</li>
            </ul>
        `;
    } else if (temp < 12) {
        return `
            <ul style="line-height: 1.8;">
                <li>Reducir frecuencia de riego para evitar encharcamientos</li>
                <li>Evitar riego en horas de baja temperatura (noche/madrugada)</li>
                <li>Verificar drenaje de suelos para prevenir anegamientos</li>
                <li>Proteger sistemas de riego de posibles heladas</li>
                <li>Asegurar que el agua no esté demasiado fría</li>
                <li>Considerar riego por goteo para aplicaciones precisas</li>
            </ul>
        `;
    } else {
        return `
            <ul style="line-height: 1.8;">
                <li>Mantener riego regular según cronograma establecido</li>
                <li>Riego profundo cada 2-3 días para estimular raíces</li>
                <li>Monitorear humedad del suelo regularmente</li>
                <li>Ajustar según tipo de cultivo y etapa fenológica</li>
                <li>Utilizar sensores de humedad para optimizar el riego</li>
                <li>Considerar la evapotranspiración del cultivo</li>
            </ul>
        `;
    }
}

function getIrrigationAdvice(temp, region) {
    if (temp > 28) {
        return `
            <ul style="line-height: 1.8;">
                <li>Aumentar frecuencia de riego en 30-40%</li>
                <li>Regar temprano en la mañana (5-7am) o al atardecer</li>
                <li>Aplicar mulch para conservar humedad</li>
                <li>Monitorear signos de estrés hídrico</li>
            </ul>
        `;
    } else if (temp < 12) {
        return `
            <ul style="line-height: 1.8;">
                <li>Reducir frecuencia de riego</li>
                <li>Evitar encharcamientos que favorezcan heladas</li>
                <li>Regar solo en horas de mayor temperatura</li>
                <li>Verificar drenaje de suelos</li>
            </ul>
        `;
    } else {
        return `
            <ul style="line-height: 1.8;">
                <li>Mantener riego regular según cronograma</li>
                <li>Riego profundo cada 2-3 días</li>
                <li>Monitorear humedad del suelo</li>
                <li>Ajustar según tipo de cultivo</li>
            </ul>
        `;
    }
}

// ========================================
// NUEVA FUNCIÓN: CALIDAD DEL AIRE Y UV
// ========================================
function updateAirQualityPanel(temperature, region, month) {
    const panel = document.getElementById('airQualityPanel');
    if (!panel) return;
    
    panel.style.display = 'block';
    
    // Simular AQI (Air Quality Index) basado en temperatura y región
    let aqi = 50; // Base buena
    
    if (temperature > 28) aqi += 30; // Calor aumenta contaminación
    if (region === 'lima') aqi += 20; // Ciudad más contaminada
    if (region === 'cusco') aqi += 10; // Altitud afecta
    if (month >= 5 && month <= 9) aqi -= 10; // Invierno mejor en costa
    
    aqi = Math.min(Math.max(aqi, 0), 200);
    
    // Calcular UV Index
    let uvIndex = 5; // Base moderado
    
    if (month >= 11 || month <= 3) uvIndex += 5; // Verano
    if (region === 'cusco' || region === 'arequipa') uvIndex += 3; // Altitud
    if (region === 'iquitos') uvIndex += 2; // Cerca del ecuador
    if (temperature > 25) uvIndex += 2;
    
    uvIndex = Math.min(Math.max(uvIndex, 0), 15);
    
    // Colores y etiquetas
    let aqiColor, aqiLabel, aqiAdvice;
    if (aqi <= 50) {
        aqiColor = '#4caf50';
        aqiLabel = 'Buena';
        aqiAdvice = 'Calidad del aire satisfactoria. Ideal para actividades al aire libre.';
    } else if (aqi <= 100) {
        aqiColor = '#ffc107';
        aqiLabel = 'Moderada';
        aqiAdvice = 'Calidad del aire aceptable para la mayoría de las personas.';
    } else if (aqi <= 150) {
        aqiColor = '#ff9800';
        aqiLabel = 'Dañina (sensibles)';
        aqiAdvice = 'Grupos sensibles deben limitar actividades prolongadas al aire libre.';
    } else {
        aqiColor = '#f44336';
        aqiLabel = 'Dañina';
        aqiAdvice = 'Todos deben limitar actividades al aire libre. Usar mascarilla recomendada.';
    }
    
    let uvColor, uvLabel, uvAdvice;
    if (uvIndex <= 2) {
        uvColor = '#4caf50';
        uvLabel = 'Bajo';
        uvAdvice = 'Protección solar no necesaria. Disfrutar del sol con precaución.';
    } else if (uvIndex <= 5) {
        uvColor = '#ffc107';
        uvLabel = 'Moderado';
        uvAdvice = 'Protección necesaria al mediodía. Usar sombrero y gafas de sol.';
    } else if (uvIndex <= 7) {
        uvColor = '#ff9800';
        uvLabel = 'Alto';
        uvAdvice = 'Protección necesaria - usar FPS 30+, sombrero y gafas de sol.';
    } else if (uvIndex <= 10) {
        uvColor = '#f44336';
        uvLabel = 'Muy Alto';
        uvAdvice = 'Protección extra - FPS 50+, sombrero, gafas y buscar sombra.';
    } else {
        uvColor = '#9c27b0';
        uvLabel = 'Extremo';
        uvAdvice = 'Evitar exposición solar entre 10am-4pm. Máxima protección obligatoria.';
    }
    
    // Recomendaciones adicionales basadas en la región
    let regionalAdvice = '';
    if (region === 'lima') {
        regionalAdvice = '<p><strong>📍 Lima:</strong> La contaminación vehicular es la principal fuente de mala calidad del aire. Evitar zonas de alto tráfico en horas pico (7-9am y 6-8pm).</p>';
    } else if (region === 'cusco') {
        regionalAdvice = '<p><strong>📍 Cusco:</strong> La altitud aumenta la radiación UV. Usar protección solar incluso en días nublados. El aire es más limpio pero la radiación es más intensa.</p>';
    } else if (region === 'iquitos') {
        regionalAdvice = '<p><strong>📍 Iquitos:</strong> Alta humedad y calor pueden empeorar la calidad del aire. Mantener buena ventilación en interiores. Usar repelente contra mosquitos.</p>';
    } else if (region === 'arequipa') {
        regionalAdvice = '<p><strong>📍 Arequipa:</strong> Baja humedad y alta radiación solar. Hidratación constante es esencial. El aire es generalmente limpio pero seco.</p>';
    } else if (region === 'trujillo') {
        regionalAdvice = '<p><strong>📍 Trujillo:</strong> Condiciones similares a Lima pero con menor contaminación. Precaución con la radiación solar durante todo el año.</p>';
    }
    
    // Recomendaciones por mes
    let monthlyAdvice = '';
    const monthNames = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                       'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'];
    
    if (month >= 12 || month <= 3) {
        monthlyAdvice = `<p><strong>📅 ${monthNames[month-1]}:</strong> Temporada de verano. Máxima radiación UV. Mayor riesgo de insolación y deshidratación.</p>`;
    } else if (month >= 6 && month <= 8) {
        monthlyAdvice = `<p><strong>📅 ${monthNames[month-1]}:</strong> Temporada de invierno. Menor radiación UV pero mayor riesgo de enfermedades respiratorias.</p>`;
    }
    
    document.getElementById('aqiIndicator').innerHTML = `
        <div class="aqi-value" style="color: ${aqiColor}; font-size: 2.5em; font-weight: bold;">${Math.round(aqi)}</div>
        <div class="aqi-label" style="color: ${aqiColor};">${aqiLabel}</div>
    `;
    
    document.getElementById('uvIndicator').innerHTML = `
        <div class="uv-value" style="color: ${uvColor}; font-size: 2.5em; font-weight: bold;">${Math.round(uvIndex)}</div>
        <div class="uv-label" style="color: ${uvColor};">${uvLabel}</div>
    `;
    
    document.getElementById('qualityAdvice').innerHTML = `
        <div style="margin-top: 15px;">
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 15px;">
                <p style="margin: 10px 0;"><strong>Calidad del Aire:</strong> ${aqiAdvice}</p>
                <p style="margin: 10px 0;"><strong>Radiación UV:</strong> ${uvAdvice}</p>
                ${regionalAdvice}
                ${monthlyAdvice}
            </div>
            
            <div style="background: rgba(255,255,255,0.05); border-radius: 8px; padding: 15px;">
                <h4 style="color: #ff6b6b; margin-top: 0;">📊 Índices Explicados:</h4>
                <p><strong>AQI (Índice de Calidad del Aire):</strong> Mide la concentración de contaminantes. Valores de 0-50 son buenos, 51-100 moderados, 101-150 poco saludables para grupos sensibles, y más de 150 poco saludables para todos.</p>
                <p><strong>Índice UV:</strong> Mide la intensidad de la radiación ultravioleta. Valores de 0-2 son bajos, 3-5 moderados, 6-7 altos, 8-10 muy altos, y 11+ extremos.</p>
            </div>
        </div>
    `;
}

// ========================================
// FUNCIONES DE GRÁFICOS Y MAPAS
// ========================================
function updatePredictionChart(year, temperature, region) {
    const historical = historicalData[region];
    if (!historical || !historical.years || historical.years.length === 0) {
        console.warn('No hay datos historicos para', region);
        return;
    }
    
    const lastHistoricalYear = Math.max(...historical.years);
    const firstHistoricalYear = Math.min(...historical.years);
    
    let allYears = [...historical.years];
    let historicalTemps = [...historical.temperatures];
    let predictionTemps = new Array(historical.temperatures.length).fill(null);
    
    if (year > lastHistoricalYear) {
        const lastTemp = historical.temperatures[historical.temperatures.length - 1];
        const yearsToPredict = year - lastHistoricalYear;
        
        // OPTIMIZACION: Limitar puntos intermedios
        const maxIntermediatePoints = 10;
        const step = Math.max(1, Math.floor(yearsToPredict / maxIntermediatePoints));
        
        for (let i = step; i <= yearsToPredict; i += step) {
            const intermediateYear = lastHistoricalYear + i;
            const progress = i / yearsToPredict;
            const intermediateTemp = lastTemp + (temperature - lastTemp) * Math.pow(progress, 0.8);
            
            allYears.push(intermediateYear);
            historicalTemps.push(null);
            predictionTemps.push(parseFloat(intermediateTemp.toFixed(1)));
        }
        
        // Asegurar que el año objetivo esté incluido
        if (allYears[allYears.length - 1] !== year) {
            allYears.push(year);
            historicalTemps.push(null);
            predictionTemps.push(temperature);
        }
        
        // Conectar último histórico con primera predicción
        predictionTemps[predictionTemps.length - allYears.filter(y => y > lastHistoricalYear).length - 1] = lastTemp;
        
    } else if (year >= firstHistoricalYear && year <= lastHistoricalYear) {
        const yearIndex = historical.years.indexOf(year);
        if (yearIndex !== -1) {
            predictionTemps[yearIndex] = temperature;
        }
    }
    
    // Actualizar gráfico
    predictionChart.data.labels = allYears;
    predictionChart.data.datasets[0].data = historicalTemps;
    predictionChart.data.datasets[1].data = predictionTemps;
    
    // Ajustar rango Y
    const allTemps = [...historicalTemps.filter(t => t !== null), 
                      ...predictionTemps.filter(t => t !== null)];
    const minTemp = Math.min(...allTemps);
    const maxTemp = Math.max(...allTemps);
    const padding = (maxTemp - minTemp) * 0.15;
    
    predictionChart.options.scales.y.min = Math.floor(minTemp - padding);
    predictionChart.options.scales.y.max = Math.ceil(maxTemp + padding);
    
    predictionChart.update('active');
}

// Función auxiliar para calcular la tendencia de calentamiento
function calculateWarmingTrend(years, temperatures) {
    if (years.length < 2) return 0.03; // Valor por defecto
    
    // Calcular la tendencia lineal usando regresión simple
    const n = years.length;
    const sumX = years.reduce((a, b) => a + b, 0);
    const sumY = temperatures.reduce((a, b) => a + b, 0);
    const sumXY = years.reduce((sum, xi, i) => sum + xi * temperatures[i], 0);
    const sumXX = years.reduce((sum, xi) => sum + xi * xi, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope; 
}

function initializeMap() {
    if (map) return;
    
    try {
        map = L.map('mapContainer').setView([-9.1900, -75.0152], 6);
        
        L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
            attribution: 'OpenStreetMap contributors',
            maxZoom: 15
        }).addTo(map);
        
        showMessage('Mapa inicializado', 'success');
        updateMapPredictions();
    } catch (error) {
        console.error('Error mapa:', error);
        showNotification('Error al cargar mapa', 'error');
    }
}

async function updateMapPredictions() {
    if (!map) return;
    
    const year = parseInt(document.getElementById('year').value) || 2030;
    const month = parseInt(document.getElementById('month').value) || 6;
    
    try {
        const response = await fetch(`${API_BASE}/predict_map?year=${year}&model=${currentModel}&month=${month}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayMapPredictions(data.predictions);
            updateMapStatistics(data.predictions, data.statistics);
            showMessage(`Mapa actualizado: ${data.month_name} ${year}`, 'success');
        }
    } catch (error) {
        console.error('Error:', error);
        displayFallbackMapData(year, month);
    }
}

function displayMapPredictions(predictions) {
    map.eachLayer(layer => {
        if (layer instanceof L.Marker || layer instanceof L.CircleMarker) {
            map.removeLayer(layer);
        }
    });
    
    predictions.forEach(pred => {
        const color = getTemperatureColor(pred.temperature);
        
        const marker = L.circleMarker([pred.latitude, pred.longitude], {
            radius: Math.max(8, Math.min(20, pred.confidence / 5)),
            fillColor: color,
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(map);
        
        const popupContent = `
            <div style="text-align: center; min-width: 200px; color: white;">
                <h4 style="color: #4ecdc4; margin: 0 0 10px 0;">${pred.city}</h4>
                <p><strong>Temp:</strong> ${pred.temperature}°C</p>
                <p><strong>Confianza:</strong> ${pred.confidence}%</p>
                <p><strong>Zona:</strong> ${pred.climate_zone || 'N/A'}</p>
                <p><strong>Alerta:</strong> <span style="color: ${color};">${pred.alert_level.replace('_', ' ')}</span></p>
                <div style="margin-top: 10px;">
                    <strong>Detalles:</strong><br>
                    Altitud: ${pred.altitude}m<br>
                    Población: ${pred.population?.toLocaleString() || 'N/A'}
                </div>
            </div>
        `;
        
        marker.bindPopup(popupContent);
        marker.on('mouseover', function() { this.openPopup(); });
        marker.on('mouseout', function() { this.closePopup(); });
    });
    
    // Ajustar vista del mapa para mostrar todos los marcadores
    if (predictions.length > 0) {
        const group = new L.featureGroup(
            predictions.map(pred => 
                L.marker([pred.latitude, pred.longitude])
            )
        );
        map.fitBounds(group.getBounds().pad(0.1));
    }
}

function getTemperatureColor(temp) {
    if (temp < 10) return '#3f51b5';      // Azul oscuro - Frío extremo
    if (temp < 15) return '#2196f3';      // Azul - Frío
    if (temp < 20) return '#4ecdc4';      // Turquesa - Fresco
    if (temp < 25) return '#96ceb4';      // Verde - Normal
    if (temp < 28) return '#ffc107';      // Amarillo - Cálido
    if (temp < 32) return '#ff9800';      // Naranja - Caliente
    return '#f44336';                     // Rojo - Calor extremo
}

function displayFallbackMapData(year, month) {
    const fallbackData = [
        { city: 'Lima', latitude: -12.0464, longitude: -77.0428, temperature: 19.5, confidence: 75 },
        { city: 'Cusco', latitude: -13.5319, longitude: -71.9675, temperature: 15.2, confidence: 75 },
        { city: 'Arequipa', latitude: -16.4090, longitude: -71.5375, temperature: 18.3, confidence: 75 },
        { city: 'Iquitos', latitude: -3.7437, longitude: -73.2516, temperature: 26.1, confidence: 75 },
        { city: 'Trujillo', latitude: -8.1116, longitude: -79.0290, temperature: 22.4, confidence: 75 }
    ];
    
    displayMapPredictions(fallbackData);
    updateMapStatistics(fallbackData, { avg_temperature: 20.3, max_temperature: 26.1, min_temperature: 15.2, cities_high_alert: 1 });
    showMessage('Usando datos de respaldo', 'warning');
}

function updateMapStatistics(predictions, statistics) {
    if (statistics) {
        document.getElementById('avgTemp').textContent = statistics.avg_temperature.toFixed(1) + '°C';
        document.getElementById('maxTemp').textContent = statistics.max_temperature.toFixed(1) + '°C';
        document.getElementById('minTemp').textContent = statistics.min_temperature.toFixed(1) + '°C';
        document.getElementById('alertCount').textContent = (statistics.cities_high_alert + statistics.cities_low_alert) || 0;
    } else {
        const temps = predictions.map(p => p.temperature);
        document.getElementById('avgTemp').textContent = (temps.reduce((a,b) => a+b, 0) / temps.length).toFixed(1) + '°C';
        document.getElementById('maxTemp').textContent = Math.max(...temps).toFixed(1) + '°C';
        document.getElementById('minTemp').textContent = Math.min(...temps).toFixed(1) + '°C';
        document.getElementById('alertCount').textContent = predictions.filter(p => p.alert_level && !['normal'].includes(p.alert_level)).length;
    }
}

function getTemperatureColor(temp) {
    if (temp < 10) return '#3f51b5';
    if (temp < 15) return '#2196f3';
    if (temp < 20) return '#4ecdc4';
    if (temp < 25) return '#96ceb4';
    if (temp < 28) return '#ffc107';
    if (temp < 32) return '#ff9800';
    return '#f44336';
}

// ========================================
// COMPARACIÓN DE REGIONES
// ========================================
async function compareRegions() {
    const year = parseInt(document.getElementById('year').value) || 2030;
    const month = parseInt(document.getElementById('month').value) || 6;
    const regions = ['lima', 'cusco', 'arequipa', 'iquitos', 'trujillo'];
    
    // AÑADIR INDICADOR VISUAL
    showNotification('🔄 Generando comparación entre regiones...', 'info');
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/comparison/${regions.join(',')}?year=${year}&month=${month}&model=${currentModel}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayComparisonResults(data);
            
            // CAMBIAR A TAB DE COMPARACIÓN AUTOMÁTICAMENTE
            document.querySelector('.tab[data-tab="comparison"]').click();
            
            showNotification('✅ Comparación completada exitosamente', 'success');
        } else {
            generateFallbackComparison(regions, year, month);
        }
    } catch (error) {
        console.error('Error:', error);
        generateFallbackComparison(regions, year, month);
        showNotification('⚠️ Usando datos de respaldo para comparación', 'warning');
    }
    
    showLoading(false);
}

function displayComparisonResults(data) {
    if (!comparisonChart || !data || !data.regions) return;

    const datasets = data.regions.map((region, index) => {
        const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffc107'];
        return {
            label: region.region_name || region.region,
            data: Array.from({length: 12}, (_, i) => {
                // Simular datos mensuales basados en la predicción
                const baseTemp = region.prediction?.temperature || 20;
                const seasonalVariation = Math.sin((i - 1) * Math.PI / 6) * 2;
                return baseTemp + seasonalVariation;
            }),
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length] + '20',
            borderWidth: 2
        };
    });
    
    comparisonChart.data.datasets = datasets;
    comparisonChart.update('active');
    
    if (data.comparison_summary) {
        document.getElementById('comparisonAnalysis').innerHTML = `
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; margin-bottom: 20px;">
                <h4 style="color: #4ecdc4; margin-top: 0;">📋 Resumen Comparativo</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div>
                        <p><strong>Más cálida:</strong> ${data.comparison_summary.warmest_region}</p>
                        <p><strong>Temperatura:</strong> ${data.comparison_summary.max_temperature}°C</p>
                    </div>
                    <div>
                        <p><strong>Más fría:</strong> ${data.comparison_summary.coolest_region}</p>
                        <p><strong>Temperatura:</strong> ${data.comparison_summary.min_temperature}°C</p>
                    </div>
                    <div>
                        <p><strong>Promedio:</strong> ${data.comparison_summary.avg_temperature}°C</p>
                        <p><strong>Rango:</strong> ${data.comparison_summary.temperature_range}°C</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    const regionCards = data.regions.map(region => {
        const temp = region.prediction?.temperature || 20;
        const confidence = region.prediction?.confidence || 75;
        const alertLevel = getTemperatureAlertLevel(temp, region.region);
        const alertColor = alertLevel === 'extreme_high' || alertLevel === 'high' ? '#ef4444' : 
                          alertLevel === 'extreme_low' || alertLevel === 'low' ? '#3b82f6' : '#10b981';
        
        return `
            <div class="region-card" style="background: rgba(255, 255, 255, 0.03); border-radius: 12px; padding: 20px; border-left: 4px solid ${alertColor};">
                <div class="region-name">${region.region_name || capitalizeFirst(region.region)}</div>
                <div class="region-stats">
                    <div>
                        <span class="stat-label">Predicción:</span>
                        <span class="stat-value" style="color: ${alertColor};">${temp}°C</span>
                    </div>
                    <div>
                        <span class="stat-label">Confianza:</span>
                        <span class="stat-value">${confidence}%</span>
                    </div>
                    <div>
                        <span class="stat-label">Alerta:</span>
                        <span class="stat-value" style="color: ${alertColor};">${alertLevel.replace('_', ' ')}</span>
                    </div>
                </div>
                <div style="margin-top: 15px; font-size: 0.9em;">
                    <p><strong>Variación estacional:</strong> ±2°C</p>
                    <p><strong>Tendencia histórica:</strong> +0.2°C/década</p>
                </div>
            </div>
        `;
    }).join('');
    
    document.getElementById('regionComparison').innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
            ${regionCards}
        </div>
    `;
}

function getTemperatureAlertLevel(temp, region) {
    // Umbrales de alerta por región (°C)
    const alertThresholds = {
        'lima': {'extreme_high': 28, 'high': 25, 'low': 15, 'extreme_low': 12},
        'cusco': {'extreme_high': 22, 'high': 20, 'low': 8, 'extreme_low': 5},
        'arequipa': {'extreme_high': 25, 'high': 22, 'low': 10, 'extreme_low': 7},
        'iquitos': {'extreme_high': 32, 'high': 30, 'low': 22, 'extreme_low': 20},
        'trujillo': {'extreme_high': 30, 'high': 28, 'low': 16, 'extreme_low': 14}
    };
    
    const thresholds = alertThresholds[region] || alertThresholds['lima'];
    
    if (temp >= thresholds['extreme_high']) return 'extreme_high';
    if (temp >= thresholds['high']) return 'high';
    if (temp <= thresholds['extreme_low']) return 'extreme_low';
    if (temp <= thresholds['low']) return 'low';
    return 'normal';
}

function generateFallbackComparison(regions, year, month) {
    const fallbackData = regions.map(region => ({
        region,
        region_name: capitalizeFirst(region),
        prediction: {
            temperature: parseFloat((getBaseTemperature(region) + (year - 2024) * 0.03).toFixed(1)),
            confidence: 75
        }
    }));
    
    displayComparisonResults({ regions: fallbackData, comparison_summary: {
        warmest_region: regions[0],
        coolest_region: regions[1],
        avg_temperature: 20
    }});
    
    showMessage('Comparación con datos de respaldo', 'warning');
}

function loadComparisonData() {
    setTimeout(() => compareRegions(), 500);
}

// ========================================
// DESCARGA DE CSV
// ========================================
async function downloadPredictions() {
    const year = parseInt(document.getElementById('year').value) || 2030;
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/download_predictions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                year,
                model: currentModel,
                regions: ['peru', 'lima', 'cusco', 'arequipa', 'iquitos', 'trujillo'],
                months: Array.from({length: 12}, (_, i) => i + 1)
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            downloadCSV(data.csv_content, data.filename);
            showNotification(`CSV generado: ${data.file_stats.total_predictions} predicciones`, 'success');
        }
    } catch (error) {
        console.error('Error:', error);
        generateFallbackCSV(year);
    }
    
    showLoading(false);
}

function generateFallbackCSV(year) {
    let csvContent = "region,mes,temperatura,confianza,modelo\n";
    const regions = ['lima', 'cusco', 'arequipa', 'iquitos', 'trujillo'];
    const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
    
    regions.forEach(region => {
        months.forEach((month, idx) => {
            const temp = (getBaseTemperature(region) + Math.sin(idx * Math.PI / 6) * 2).toFixed(1);
            csvContent += `${region},${month},${temp},75,${currentModel}\n`;
        });
    });
    
    downloadCSV(csvContent, `predicciones_${year}.csv`);
    showNotification('CSV de respaldo generado', 'warning');
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Liberar memoria
    setTimeout(() => {
        URL.revokeObjectURL(url);
    }, 100);
    
    showNotification(`CSV descargado: ${filename}`, 'success');
}

function generateFallbackCSV(year) {
    let csvContent = "region,mes,temperatura,confianza,modelo\n";
    const regions = ['lima', 'cusco', 'arequipa', 'iquitos', 'trujillo'];
    const months = ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'];
    
    regions.forEach(region => {
        months.forEach((month, idx) => {
            const baseTemp = getBaseTemperature(region);
            const seasonalVariation = Math.sin(idx * Math.PI / 6) * 2;
            const temp = (baseTemp + seasonalVariation).toFixed(1);
            csvContent += `${region},${month},${temp},75,${currentModel}\n`;
        });
    });
    
    downloadCSV(csvContent, `predicciones_${year}.csv`);
    showNotification('CSV de respaldo generado', 'warning');
}

// ========================================
// ANÁLISIS HISTÓRICO
// ========================================
function updateHistoricalAnalysis() {
    const region = document.getElementById('region').value;
    const data = historicalData[region];
    
    if (data && data.statistics) {
        // Calcular tendencias adicionales
        const tempTrend = data.statistics.warming_rate_per_decade || 0.2;
        const trendDirection = tempTrend > 0 ? 'calentamiento' : 'enfriamiento';
        const trendStrength = Math.abs(tempTrend) > 0.3 ? 'significativo' : 'moderado';
        
        // Calcular años extremos
        const temps = data.temperatures;
        const years = data.years;
        const maxTempIndex = temps.indexOf(Math.max(...temps));
        const minTempIndex = temps.indexOf(Math.min(...temps));
        
        document.getElementById('historicalAnalysis').innerHTML = `
            <div style="background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px;">
                <h4 style="color: #4ecdc4; margin-top: 0;">Análisis de ${data.region_display_name}</h4>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                    <div>
                        <h5 style="color: #ff6b6b;">📊 Estadísticas Básicas</h5>
                        <p><strong>Temperatura promedio:</strong> ${data.statistics.mean_temperature?.toFixed(1) || 'N/A'}°C</p>
                        <p><strong>Tendencia:</strong> ${trendStrength} ${trendDirection} (${tempTrend.toFixed(2)}°C/década)</p>
                        <p><strong>Datos:</strong> ${data.years?.length || 0} años de registros</p>
                    </div>
                    
                    <div>
                        <h5 style="color: #96ceb4;">🌡️ Extremos Históricos</h5>
                        <p><strong>Máxima:</strong> ${Math.max(...temps).toFixed(1)}°C (${years[maxTempIndex]})</p>
                        <p><strong>Mínima:</strong> ${Math.min(...temps).toFixed(1)}°C (${years[minTempIndex]})</p>
                        <p><strong>Rango:</strong> ${(Math.max(...temps) - Math.min(...temps)).toFixed(1)}°C</p>
                    </div>
                    
                    <div>
                        <h5 style="color: #ffc107;">📈 Variabilidad</h5>
                        <p><strong>Desviación estándar:</strong> ${data.statistics.std_deviation?.toFixed(1) || 'N/A'}°C</p>
                        <p><strong>Coeficiente de variación:</strong> ${((data.statistics.std_deviation / data.statistics.mean_temperature) * 100).toFixed(1)}%</p>
                        <p><strong>Estabilidad:</strong> ${data.statistics.std_deviation < 1.5 ? 'Alta' : data.statistics.std_deviation < 3 ? 'Moderada' : 'Baja'}</p>
                    </div>
                </div>
                
                <div style="margin-top: 25px;">
                    <h5 style="color: #45b7d1;">📊 Proyecciones Climáticas</h5>
                    <div style="background: rgba(69, 183, 209, 0.1); border-radius: 10px; padding: 15px; margin-top: 10px;">
                        <p><strong>Escenario 2030:</strong> ${(data.statistics.mean_temperature + (2030 - 2024) * tempTrend / 10).toFixed(1)}°C</p>
                        <p><strong>Escenario 2050:</strong> ${(data.statistics.mean_temperature + (2050 - 2024) * tempTrend / 10).toFixed(1)}°C</p>
                        <p><strong>Escenario 2100:</strong> ${(data.statistics.mean_temperature + (2100 - 2024) * tempTrend / 10).toFixed(1)}°C</p>
                    </div>
                </div>
                
                <div style="margin-top: 25px;">
                    <h5 style="color: #f59e0b;">⚠️ Eventos Extremos Registrados</h5>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 10px;">
                        ${getExtremeEventsForRegion(region).map(event => `
                            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 8px; padding: 12px;">
                                <p style="margin: 0; font-weight: bold;">${event.year}</p>
                                <p style="margin: 5px 0 0 0; font-size: 0.9em;">${event.event}</p>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    } else {
        document.getElementById('historicalAnalysis').innerHTML = `
            <div style="background: rgba(255, 107, 107, 0.1); border-radius: 15px; padding: 25px; text-align: center;">
                <h4 style="color: #ff6b6b; margin-top: 0;">⚠️ Datos No Disponibles</h4>
                <p>No se encontraron datos históricos para la región seleccionada.</p>
                <p>Por favor, seleccione otra región o verifique la conexión.</p>
            </div>
        `;
    }
}

// Función auxiliar para obtener eventos extremos por región
function getExtremeEventsForRegion(region) {
    const eventsData = {
        lima: [
            { year: '2017', event: 'Ola de calor' },
            { year: '2017', event: 'Lluvias intensas' },
            { year: '2018', event: 'Helada atípica' },
            { year: '2020', event: 'Temperatura récord' }
        ],
        cusco: [
            { year: '2003', event: 'Heladas severas' },
            { year: '2013', event: 'Heladas' },
            { year: '2016', event: 'Sequía' },
            { year: '2021', event: 'Lluvias torrenciales' }
        ],
        arequipa: [
            { year: '2007', event: 'Heladas' },
            { year: '2015', event: 'Sequía' },
            { year: '2019', event: 'Lluvias intensas' },
            { year: '2022', event: 'Ola de calor' }
        ],
        iquitos: [
            { year: '2012', event: 'Inundaciones' },
            { year: '2015', event: 'Inundaciones' },
            { year: '2016', event: 'Sequía' },
            { year: '2020', event: 'Nivel récord de ríos' }
        ],
        trujillo: [
            { year: '2014', event: 'Sequía' },
            { year: '2016', event: 'Ola de calor' },
            { year: '2017', event: 'Lluvias intensas' },
            { year: '2021', event: 'Temperatura extrema' }
        ],
        peru: [
            { year: '1997-98', event: 'Fenómeno El Niño' },
            { year: '2015-16', event: 'Fenómeno El Niño' },
            { year: '2013', event: 'Heladas nacionales' },
            { year: '2017', event: 'Eventos extremos múltiples' }
        ]
    };
    
    return eventsData[region] || eventsData.peru;
}

function updateMap() {
    if (map) setTimeout(() => updateMapPredictions(), 100);
}

function updateMonthlyTrends() {
    if (!temperatureTrendChart) return;
    
    const region = document.getElementById('region').value;
    const year = parseInt(document.getElementById('year').value) || 2030;
    const months = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
    const baseTemp = getBaseTemperature(region);
    
    const monthlyData = months.map((_, idx) => {
        return parseFloat((baseTemp + Math.sin(idx * Math.PI / 6) * 2 + (year - 2024) * 0.03).toFixed(1));
    });
    
    temperatureTrendChart.data.labels = months;
    temperatureTrendChart.data.datasets[0].data = monthlyData;
    temperatureTrendChart.update('active');
}

// ========================================
// UTILIDADES
// ========================================
function capitalizeFirst(str) {
    if (!str || typeof str !== 'string') return str;
    return str.charAt(0).toUpperCase() + str.slice(1).replace('_', ' ');
}

function showLoading(show) {
    const loadingElement = document.getElementById('loading');
    if (loadingElement) {
        loadingElement.style.display = show ? 'flex' : 'none';
    }
    
    // Deshabilitar botones durante carga
    const buttons = document.querySelectorAll('.predict-btn, .download-btn, .compare-btn');
    buttons.forEach(btn => {
        btn.disabled = show;
        btn.style.opacity = show ? '0.6' : '1';
    });
}

function showMessage(message, type = 'success') {
    const container = document.getElementById('messageContainer');
    if (!container) return;
    
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = message;
    
    container.innerHTML = '';
    container.appendChild(div);
    
    setTimeout(() => {
        if (div.parentNode) {
            div.remove();
        }
    }, 5000);
}

function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    if (!notification) return;
    
    notification.className = `notification ${type} show`;
    notification.textContent = message;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 4000);
}

function getBaseTemperature(region) {
    const baseTemps = {
        'peru': 18.5, 'lima': 19.0, 'cusco': 15.0,
        'arequipa': 18.0, 'iquitos': 26.0, 'trujillo': 22.0
    };
    return baseTemps[region] || 18.5;
}

function getClimateContext(temperature, region) {
    if (temperature > 30) return "Temperatura excepcionalmente alta - Posible ola de calor";
    if (temperature < 10) return "Temperatura muy baja - Condiciones frías extremas";
    if (temperature > 25) return "Temperatura cálida - Por encima del promedio";
    if (temperature < 15) return "Temperatura fresca - Por debajo del promedio";
    return "Temperatura dentro del rango esperado";
}

function getAlertLevel(temp, region) {
    if (temp > 30) return 'extreme_high';
    if (temp > 27) return 'high';
    if (temp < 8) return 'extreme_low';
    if (temp < 12) return 'low';
    return 'normal';
}

function animateTemperature(element, targetTemp) {
    const currentTemp = parseFloat(element.textContent) || 0;
    const duration = 2000;
    const steps = 60;
    const increment = (targetTemp - currentTemp) / steps;
    let step = 0;
    
    const animation = setInterval(() => {
        step++;
        element.textContent = (currentTemp + increment * step).toFixed(1) + '°C';
        if (step >= steps) {
            clearInterval(animation);
            element.textContent = targetTemp.toFixed(1) + '°C';
        }
    }, duration / steps);
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
    document.querySelectorAll('.predict-btn, .download-btn, .compare-btn').forEach(btn => {
        btn.disabled = show;
        btn.style.opacity = show ? '0.6' : '1';
    });
}

function showMessage(message, type = 'success') {
    const container = document.getElementById('messageContainer');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    div.textContent = message;
    container.innerHTML = '';
    container.appendChild(div);
    setTimeout(() => div.remove(), 5000);
}

function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    notification.className = `notification ${type} show`;
    notification.textContent = message;
    setTimeout(() => notification.classList.remove('show'), 4000);
}
// Inicializar trends
setTimeout(() => { if (temperatureTrendChart) updateMonthlyTrends(); }, 3000);

// ========================================
// EVENTOS EXTREMOS
// ========================================
async function loadExtremeEvents() {
    const year = parseInt(document.getElementById('year').value);
    const region = document.getElementById('region').value;
    
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE}/extreme_events?year=${year}&region=${region}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            displayExtremeEvents(data);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error al cargar eventos extremos', 'error');
    } finally {
        showLoading(false);
    }
}
function displayExtremeEvents(data) {
    const container = document.getElementById('extremeEventsContainer');
    
    let html = '<div class="events-grid">';
    
    data.extreme_events.forEach(event => {
        const riskScore = data.risk_assessment.find(r => r.event === event.event);
        const riskClass = riskScore?.risk_level === 'Crítico' ? 'high' : 
                          riskScore?.risk_level === 'Alto' ? 'high' : 'medium';
        
        html += `
            <div class="event-card">
                <h3>${event.event}</h3>
                <div class="risk-indicator risk-${riskClass}">
                    ${riskScore?.risk_level || 'Evaluando...'}
                </div>
                <p><strong>Probabilidad:</strong> ${event.probability}</p>
                <p><strong>Severidad:</strong> ${event.severity}</p>
                <p><strong>Duración:</strong> ${event.duration}</p>
                <h4>Impactos Potenciales:</h4>
                <ul class="impact-list">
                    ${event.impacts.map(impact => `<li>${impact}</li>`).join('')}
                </ul>
            </div>
        `;
    });
    
    html += '</div>';
    
    // Acciones de preparación
    html += `
        <div class="preparedness-section" style="margin-top: 30px; padding: 25px; background: rgba(16, 185, 129, 0.1); border-radius: 15px; border: 2px solid #10b981;">
            <h3 style="color: #10b981;">🛡️ Acciones de Preparación</h3>
            <div class="actions-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>⚡ Inmediatas</h4>
                    <ul>${data.preparedness_actions.immediate.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
                <div>
                    <h4>📅 Corto Plazo</h4>
                    <ul>${data.preparedness_actions.short_term.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
                <div>
                    <h4>🎯 Largo Plazo</h4>
                    <ul>${data.preparedness_actions.long_term.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
            </div>
        </div>
    `;
    
    // Sección de recursos adicionales
    html += `
        <div class="resources-section" style="margin-top: 30px; padding: 25px; background: rgba(59, 130, 246, 0.1); border-radius: 15px; border: 2px solid #3b82f6;">
            <h3 style="color: #3b82f6;">📚 Recursos de Información</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>🌐 Sitios Web Oficiales</h4>
                    <ul>
                        <li><a href="https://www.senamhi.gob.pe/" target="_blank">SENAMHI - Servicio Nacional de Meteorología</a></li>
                        <li><a href="https://www.indeci.gob.pe/" target="_blank">INDECI - Instituto Nacional de Defensa Civil</a></li>
                        <li><a href="https://www.minam.gob.pe/" target="_blank">MINAM - Ministerio del Ambiente</a></li>
                    </ul>
                </div>
                <div>
                    <h4>📱 Aplicaciones Móviles</h4>
                    <ul>
                        <li>SENAMHI Móvil - Alertas meteorológicas</li>
                        <li>INDECI - Información de emergencias</li>
                        <li>Clima Perú - Pronósticos locales</li>
                    </ul>
                </div>
                <div>
                    <h4>📞 Líneas de Emergencia</h4>
                    <ul>
                        <li><strong>Emergencias:</strong> 116</li>
                        <li><strong>Defensa Civil:</strong> 115</li>
                        <li><strong>Bomberos:</strong> 116</li>
                        <li><strong>Policía:</strong> 105</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}
function displayExtremeEvents(data) {
    const container = document.getElementById('extremeEventsContainer');
    
    let html = '<div class="events-grid">';
    
    data.extreme_events.forEach(event => {
        const riskScore = data.risk_assessment.find(r => r.event === event.event);
        const riskClass = riskScore?.risk_level === 'Crítico' ? 'high' : 
                          riskScore?.risk_level === 'Alto' ? 'high' : 'medium';
        
        html += `
            <div class="event-card">
                <h3>${event.event}</h3>
                <div class="risk-indicator risk-${riskClass}">
                    ${riskScore?.risk_level || 'Evaluando...'}
                </div>
                <p><strong>Probabilidad:</strong> ${event.probability}</p>
                <p><strong>Severidad:</strong> ${event.severity}</p>
                <p><strong>Duración:</strong> ${event.duration}</p>
                <h4>Impactos Potenciales:</h4>
                <ul class="impact-list">
                    ${event.impacts.map(impact => `<li>${impact}</li>`).join('')}
                </ul>
            </div>
        `;
    });
    
    html += '</div>';
    
    // Acciones de preparación
    html += `
        <div class="preparedness-section" style="margin-top: 30px; padding: 25px; background: rgba(16, 185, 129, 0.1); border-radius: 15px; border: 2px solid #10b981;">
            <h3 style="color: #10b981;">🛡️ Acciones de Preparación</h3>
            <div class="actions-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>⚡ Inmediatas</h4>
                    <ul>${data.preparedness_actions.immediate.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
                <div>
                    <h4>📅 Corto Plazo</h4>
                    <ul>${data.preparedness_actions.short_term.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
                <div>
                    <h4>🎯 Largo Plazo</h4>
                    <ul>${data.preparedness_actions.long_term.map(action => `<li>${action}</li>`).join('')}</ul>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ========================================
// RECURSOS HÍDRICOS
// ========================================
async function loadWaterAnalysis() {
    const year = parseInt(document.getElementById('year').value);
    const region = document.getElementById('region').value;
    
    const features = createFeatureVector(year, 6, region);
    
    showLoading(true);
    
    try {
        const predResponse = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features, model: currentModel, region })
        });
        
        const prediction = await predResponse.json();
        
        const waterResponse = await fetch(`${API_BASE}/water_resources`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temperature: prediction.predicted_temperature,
                region: region,
                year: year
            })
        });
        
        const data = await waterResponse.json();
        
        if (data.status === 'success') {
            displayWaterAnalysis(data);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error al analizar recursos hídricos', 'error');
    } finally {
        showLoading(false);
    }
}
async function checkModelsStatus() {
    try {
        const response = await fetch(`${API_BASE}/debug/models`);
        const data = await response.json();
        
        console.log('📊 Estado de los modelos:', data);
        
        let message = '🔍 DIAGNÓSTICO DE MODELOS:\n\n';
        
        for (const [name, status] of Object.entries(data.models)) {
            const icon = status.is_real_model ? '✅' : '⚠️';
            message += `${icon} ${name}: ${status.is_real_model ? 'MODELO REAL' : 'FALLBACK'}\n`;
            message += `   Tipo: ${status.type}\n`;
            message += `   Scaler: ${status.has_scaler ? 'Sí' : 'No'}\n\n`;
        }
        
        message += `\n📂 Directorio: ${data.model_directory_checked}`;
        message += `\n📊 Total de modelos: ${data.total_models}`;
        
        alert(message);
        
    } catch (error) {
        console.error('Error verificando modelos:', error);
        alert('❌ Error al verificar modelos: ' + error.message);
    }
}

function displayWaterAnalysis(data) {
    const container = document.getElementById('waterAnalysisContainer');
    const analysis = data.water_analysis;
    
    let html = '<div class="water-grid">';
    
    // Glaciares
    if (analysis.glacier_melt_rate.applicable) {
        html += `
            <div class="water-card">
                <h3>🏔️ Glaciares</h3>
                <div class="big-stat" style="font-size: 3em; color: #ef4444; text-align: center; margin: 20px 0;">
                    ${analysis.glacier_melt_rate.annual_loss_percent}%
                </div>
                <p style="text-align: center;"><strong>Pérdida anual</strong></p>
                <div class="risk-indicator risk-${analysis.glacier_melt_rate.status === 'Crítico' ? 'high' : 'medium'}" style="display: block; text-align: center;">
                    ${analysis.glacier_melt_rate.status}
                </div>
                <h4>Impactos:</h4>
                <ul class="impact-list">
                    ${analysis.glacier_melt_rate.impacts.map(impact => `<li>${impact}</li>`).join('')}
                </ul>
                <div style="margin-top: 15px; padding: 10px; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
                    <p><strong>Glaciares afectados:</strong> ${analysis.glacier_melt_rate.affected_glaciers.join(', ')}</p>
                </div>
            </div>
        `;
    }
    
    // Disponibilidad de agua
    html += `
        <div class="water-card">
            <h3>💧 Disponibilidad de Agua</h3>
            <div class="big-stat" style="font-size: 2.5em; color: #3b82f6; text-align: center; margin: 20px 0;">
                ${analysis.water_availability.per_capita_m3} m³
            </div>
            <p style="text-align: center;"><strong>Per cápita/año</strong></p>
            <div class="risk-indicator risk-${analysis.water_availability.alert_level}" style="display: block; text-align: center;">
                ${analysis.water_availability.status}
            </div>
            <p style="margin-top: 15px;"><strong>Tendencia:</strong> ${analysis.water_availability.trend}</p>
            <div style="margin-top: 15px;">
                <h4>Recomendaciones:</h4>
                <ul class="impact-list">
                    ${analysis.water_availability.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Riesgo de sequía
    html += `
        <div class="water-card">
            <h3>🏜️ Riesgo de Sequía</h3>
            <div class="big-stat" style="font-size: 2.5em; color: #f59e0b; text-align: center; margin: 20px 0;">
                ${analysis.drought_risk.risk_percentage}
            </div>
            <div class="risk-indicator risk-${analysis.drought_risk.risk_level.toLowerCase()}" style="display: block; text-align: center;">
                Riesgo ${analysis.drought_risk.risk_level}
            </div>
            <h4>Sectores Vulnerables:</h4>
            <ul class="impact-list">
                ${analysis.drought_risk.vulnerable_sectors.map(sector => `<li>${sector}</li>`).join('')}
            </ul>
            <div style="margin-top: 15px;">
                <h4>Señales de Alerta Temprana:</h4>
                <ul class="impact-list">
                    ${analysis.drought_risk.early_warning_signs.map(sign => `<li>${sign}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Riesgo de inundación
    html += `
        <div class="water-card">
            <h3>🌊 Riesgo de Inundación</h3>
            <div class="big-stat" style="font-size: 2.5em; color: #3b82f6; text-align: center; margin: 20px 0;">
                ${analysis.flood_risk.risk_percentage}
            </div>
            <div class="risk-indicator risk-${analysis.flood_risk.risk_level.toLowerCase()}" style="display: block; text-align: center;">
                Riesgo ${analysis.flood_risk.risk_level}
            </div>
            <h4>Medidas de Prevención:</h4>
            <ul class="impact-list">
                ${analysis.flood_risk.prevention_measures.map(measure => `<li>${measure}</li>`).join('')}
            </ul>
            <div style="margin-top: 15px;">
                <h4>Áreas Vulnerables:</h4>
                <ul class="impact-list">
                    ${analysis.flood_risk.vulnerable_areas.map(area => `<li>${area}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    html += '</div>';
    
    // Sección de gestión integrada de recursos hídricos
    html += `
        <div class="water-management-section" style="margin-top: 30px; padding: 25px; background: rgba(16, 185, 129, 0.1); border-radius: 15px; border: 2px solid #10b981;">
            <h3 style="color: #10b981;">💧 Gestión Integrada de Recursos Hídricos</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>🏗️ Infraestructura Crítica</h4>
                    <ul>
                        <li>Presas y reservorios: 45</li>
                        <li>Plantas de tratamiento: 120</li>
                        <li>Sistemas de riego: 85</li>
                        <li>Redes de monitoreo: 200</li>
                    </ul>
                </div>
                <div>
                    <h4>📊 Indicadores Clave</h4>
                    <ul>
                        <li>Eficiencia de riego: 45%</li>
                        <li>Agua no contabilizada: 40%</li>
                        <li>Cobertura de saneamiento: 75%</li>
                        <li>Recarga de acuíferos: -2%/año</li>
                    </ul>
                </div>
                <div>
                    <h4>🎯 Metas 2030</h4>
                    <ul>
                        <li>Aumentar eficiencia a 60%</li>
                        <li>Reducir pérdidas a 25%</li>
                        <li>Alcanzar 90% cobertura</li>
                        <li>Estabilizar acuíferos</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ========================================
// IMPACTO ECONÓMICO
// ========================================
async function loadEconomicImpact() {
    const year = parseInt(document.getElementById('year').value);
    const region = document.getElementById('region').value;
    
    const features = createFeatureVector(year, 6, region);
    
    showLoading(true);
    
    try {
        const predResponse = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features, model: currentModel, region })
        });
        
        const prediction = await predResponse.json();
        
        const economicResponse = await fetch(`${API_BASE}/economic_impact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temperature: prediction.predicted_temperature,
                region: region,
                year: year
            })
        });
        
        const data = await economicResponse.json();
        
        if (data.status === 'success') {
            displayEconomicImpact(data);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error al calcular impacto económico', 'error');
    } finally {
        showLoading(false);
    }
}

function displayEconomicImpact(data) {
    const container = document.getElementById('economicImpactContainer');
    const impact = data.economic_impact;
    
    let html = '<div class="economic-grid">';
    
    // Impacto en PIB
    html += `
        <div class="economic-card" style="background: rgba(239, 68, 68, 0.1); border: 2px solid #ef4444; border-radius: 12px; padding: 25px;">
            <h3>📉 Impacto en PIB</h3>
            <div class="big-stat" style="font-size: 2.5em; color: #ef4444; text-align: center; margin: 20px 0;">
                -${impact.gdp_impact.percentage_loss}%
            </div>
            <p style="text-align: center;"><strong>Pérdida Proyectada</strong></p>
            <div style="margin-top: 20px;">
                <p><strong>Pérdida estimada:</strong> $${impact.gdp_impact.estimated_loss_usd_millions.toLocaleString()} millones USD</p>
                <p><strong>Impacto per cápita:</strong> $${impact.gdp_impact.per_capita_impact_usd.toLocaleString()} USD</p>
                <p><strong>Comparación:</strong> ${impact.gdp_impact.comparison}</p>
            </div>
        </div>
    `;
    
    // Impactos sectoriales
    html += `
        <div class="economic-card" style="grid-column: span 2; border: 2px solid rgba(245, 158, 11, 0.5); border-radius: 12px; padding: 25px;">
            <h3>🏭 Impactos por Sector</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                ${impact.sector_impacts.map(sector => `
                    <div style="padding: 15px; background: rgba(255, 255, 255, 0.03); border-radius: 10px;">
                        <h4 style="color: #f59e0b; margin-top: 0;">${sector.sector}</h4>
                        <p style="font-size: 1.8em; font-weight: bold; color: #ef4444; text-align: center; margin: 10px 0;">
                            -${sector.loss_percentage}%
                        </p>
                        <p style="font-size: 0.9em;">${sector.description}</p>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    html += '</div>';
    
    // Sección de inversión en adaptación
    html += `
        <div class="adaptation-investment-section" style="margin-top: 30px; padding: 25px; background: rgba(16, 185, 129, 0.1); border-radius: 15px; border: 2px solid #10b981;">
            <h3 style="color: #10b981;">💰 Inversión en Adaptación</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>🏗️ Inversión Requerida</h4>
                    <ul>
                        <li><strong>Infraestructura hídrica:</strong> $3,000M</li>
                        <li><strong>Agricultura climáticamente inteligente:</strong> $2,500M</li>
                        <li><strong>Protección costera:</strong> $2,000M</li>
                        <li><strong>Sistemas de alerta temprana:</strong> $1,500M</li>
                        <li><strong>Restauración de ecosistemas:</strong> $1,000M</li>
                    </ul>
                </div>
                <div>
                    <h4>📈 Beneficios Esperados</h4>
                    <ul>
                        <li>Evitar pérdidas de $8,000M en infraestructura</li>
                        <li>Mantener producción alimentaria para 30M personas</li>
                        <li>Proteger 60% de zonas costeras habitadas</li>
                        <li>Reducir mortalidad por eventos extremos en 90%</li>
                        <li>Generar 250,000 empleos verdes</li>
                    </ul>
                </div>
                <div>
                    <h4>💸 Financiamiento</h4>
                    <ul>
                        <li>Gobierno nacional: 40%</li>
                        <li>Gobiernos regionales: 20%</li>
                        <li>Cooperación internacional: 25%</li>
                        <li>Sector privado: 10%</li>
                        <li>Bonos verdes: 5%</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    // Sección de impacto en empleo
    html += `
        <div class="employment-impact-section" style="margin-top: 30px; padding: 25px; background: rgba(59, 130, 246, 0.1); border-radius: 15px; border: 2px solid #3b82f6;">
            <h3 style="color: #3b82f6;">👥 Impacto en Empleo</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>⚠️ Empleos en Riesgo</h4>
                    <ul>
                        <li><strong>Agricultura:</strong> 500,000 empleos</li>
                        <li><strong>Pesca:</strong> 150,000 empleos</li>
                        <li><strong>Turismo:</strong> 200,000 empleos</li>
                        <li><strong>Construcción:</strong> 100,000 empleos</li>
                    </ul>
                </div>
                <div>
                    <h4>🌱 Nuevas Oportunidades</h4>
                    <ul>
                        <li><strong>Energías renovables:</strong> +100,000 empleos</li>
                        <li><strong>Agricultura climáticamente inteligente:</strong> +80,000 empleos</li>
                        <li><strong>Gestión de recursos hídricos:</strong> +40,000 empleos</li>
                        <li><strong>Servicios ambientales:</strong> +30,000 empleos</li>
                    </ul>
                </div>
                <div>
                    <h4>🔄 Programas de Transición</h4>
                    <ul>
                        <li>Capacitación técnica: 200,000 beneficiarios</li>
                        <li>Reconversión laboral: 150,000 trabajadores</li>
                        <li>Apoyo a emprendimientos verdes: 50,000</li>
                        <li>Becas de especialización: 25,000</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ========================================
// BIODIVERSIDAD
// ========================================
async function loadBiodiversityImpact() {
    const year = parseInt(document.getElementById('year').value);
    const region = document.getElementById('region').value;
    
    const features = createFeatureVector(year, 6, region);
    
    showLoading(true);
    
    try {
        const predResponse = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features, model: currentModel, region })
        });
        
        const prediction = await predResponse.json();
        
        const bioResponse = await fetch(`${API_BASE}/biodiversity_impact`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                temperature: prediction.predicted_temperature,
                region: region,
                year: year
            })
        });
        
        const data = await bioResponse.json();
        
        if (data.status === 'success') {
            displayBiodiversityImpact(data);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error al analizar biodiversidad', 'error');
    } finally {
        showLoading(false);
    }
}

function displayBiodiversityImpact(data) {
    const container = document.getElementById('biodiversityContainer');
    const impact = data.biodiversity_impact;
    
    let html = '<div class="biodiversity-grid">';
    
    // Especies amenazadas
    html += `
        <div class="biodiversity-card" style="grid-column: span 2; border: 2px solid rgba(239, 68, 68, 0.5); border-radius: 12px; padding: 25px;">
            <h3>🦜 Especies Amenazadas</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
                ${impact.species_at_risk.map(species => `
                    <div style="padding: 20px; background: rgba(239, 68, 68, 0.1); border-radius: 10px; border: 2px solid #ef4444;">
                        <h4 style="color: #ef4444; margin-top: 0;">${species.name}</h4>
                        <div class="risk-indicator risk-${species.threat_level === 'Crítico' || species.threat_level === 'Alto' ? 'high' : 'medium'}">
                            Amenaza ${species.threat_level}
                        </div>
                        <p style="margin-top: 15px;"><strong>Razón:</strong> ${species.reason}</p>
                        <div style="margin-top: 15px;">
                            <p><strong>Población estimada:</strong> ${species.population || 'Datos no disponibles'}</p>
                            <p><strong>Tendencia:</strong> ${species.trend || 'Desconocida'}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Pérdida de hábitat
    html += `
        <div class="biodiversity-card" style="border: 2px solid rgba(245, 158, 11, 0.5); border-radius: 12px; padding: 25px;">
            <h3>🌳 Pérdida de Hábitat</h3>
            <div class="big-stat" style="font-size: 3em; color: #ef4444; text-align: center; margin: 20px 0;">
                ${impact.habitat_loss.percentage_loss}%
            </div>
            <p style="text-align: center;"><strong>Pérdida Proyectada</strong></p>
            <h4>Causas Principales:</h4>
            <ul class="impact-list">
                ${impact.habitat_loss.primary_causes.map(cause => `<li>${cause}</li>`).join('')}
            </ul>
            <div style="margin-top: 15px;">
                <h4>Ecosistemas Más Afectados:</h4>
                <ul class="impact-list">
                    ${impact.habitat_loss.most_affected.map(ecosystem => `<li>${ecosystem}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    html += '</div>';
    
    // Sección de servicios ecosistémicos
    html += `
        <div class="ecosystem-services-section" style="margin-top: 30px; padding: 25px; background: rgba(16, 185, 129, 0.1); border-radius: 15px; border: 2px solid #10b981;">
            <h3 style="color: #10b981;">🌍 Servicios Ecosistémicos Afectados</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                ${impact.ecosystem_services.map(service => `
                    <div style="padding: 20px; background: rgba(255, 255, 255, 0.03); border-radius: 10px;">
                        <h4 style="color: #10b981; margin-top: 0;">${service.service}</h4>
                        <div class="risk-indicator risk-${service.status === 'Degradado' ? 'high' : service.status === 'En riesgo' ? 'medium' : 'low'}">
                            ${service.status}
                        </div>
                        <p style="margin-top: 15px;"><strong>Impacto:</strong> ${service.impact}</p>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    // Sección de estrategias de conservación
    html += `
        <div class="conservation-strategies-section" style="margin-top: 30px; padding: 25px; background: rgba(59, 130, 246, 0.1); border-radius: 15px; border: 2px solid #3b82f6;">
            <h3 style="color: #3b82f6;">🛡️ Estrategias de Conservación</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>⚡ Acciones Inmediatas</h4>
                    <ul>
                        ${impact.conservation_priorities.immediate_actions.map(action => `<li>${action}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <h4>📅 Estrategias a Largo Plazo</h4>
                    <ul>
                        ${impact.conservation_priorities.long_term_strategies.map(strategy => `<li>${strategy}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <h4>💰 Fuentes de Financiamiento</h4>
                    <ul>
                        ${impact.conservation_priorities.funding_sources.map(source => `<li>${source}</li>`).join('')}
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    // Sección de áreas protegidas
    html += `
        <div class="protected-areas-section" style="margin-top: 30px; padding: 25px; background: rgba(245, 158, 11, 0.1); border-radius: 15px; border: 2px solid #f59e0b;">
            <h3 style="color: #f59e0b;">🏞️ Áreas Naturales Protegidas</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>📊 Estadísticas Clave</h4>
                    <ul>
                        <li><strong>Total de ANP:</strong> 76</li>
                        <li><strong>Superficie protegida:</strong> 22 millones de ha</li>
                        <li><strong>Porcentaje del territorio:</strong> 17.1%</li>
                        <li><strong>ANP marinas:</strong> 8</li>
                    </ul>
                </div>
                <div>
                    <h4>🌟 ANP Emblemáticas</h4>
                    <ul>
                        <li><strong>Parque Nacional Manu:</strong> Biodiversidad excepcional</li>
                        <li><strong>Reserva de Biosfera del Noroeste:</strong> Ecosistemas únicos</li>
                        <li><strong>Santuario Histórico de Machu Picchu:</strong> Patrimonio cultural y natural</li>
                        <li><strong>Reserva Nacional Paracas:</strong> Ecosistemas costeros</li>
                    </ul>
                </div>
                <div>
                    <h4>🎯 Desafíos Actuales</h4>
                    <ul>
                        <li>Falta de financiamiento sostenible</li>
                        <li>Cambio en los límites de las ANP</li>
                        <li>Actividades ilegales dentro de áreas protegidas</li>
                        <li>Cambio climático y sus efectos</li>
                    </ul>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}
// Inicializar trends
setTimeout(() => { if (temperatureTrendChart) updateMonthlyTrends(); }, 3000);