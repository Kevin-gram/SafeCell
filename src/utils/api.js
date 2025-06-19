// Simulated API service with logging and enhanced location data

import logger from './logger'

/**
 * Makes a simulated API request with logging
 * @param {string} endpoint - API endpoint
 * @param {object} options - Request options
 * @returns {Promise<any>} Response data
 */
export const apiRequest = async (endpoint, options = {}) => {
  const startTime = Date.now()
  const method = options.method || 'GET'
  
  // Log API request start
  logger.logApiRequest(endpoint, method, 0, 'pending', {
    requestData: options.body,
    params: options.params
  })
  
  try {
    // Simulate network delay
    const delay = Math.random() * 500 + 500; // Random delay between 500-1000ms
    await new Promise(resolve => setTimeout(resolve, delay));
    
    // Simulate 5% chance of API error
    if (Math.random() < 0.05) {
      throw new Error('API request failed');
    }
    
    let response
    
    // Return mock response based on endpoint
    switch (endpoint) {
      case '/api/login':
        response = mockLogin(options.body);
        break
      case '/api/detection':
        response = mockDetection(options.body);
        break
      case '/api/statistics':
        response = mockStatistics(options.params);
        break
      case '/api/feedback':
        response = mockFeedback(options.body);
        break
      default:
        throw new Error('Unknown endpoint');
    }
    
    const duration = Date.now() - startTime
    
    // Log successful API request
    logger.logApiRequest(endpoint, method, duration, 'success', {
      responseData: response,
      requestData: options.body,
      params: options.params
    })
    
    return response
    
  } catch (error) {
    const duration = Date.now() - startTime
    
    // Log failed API request
    logger.logApiRequest(endpoint, method, duration, 'error', {
      error: error.message,
      requestData: options.body,
      params: options.params
    })
    
    throw error
  }
};

/**
 * Mock login API
 */
const mockLogin = (data) => {
  const { email, password } = data || {};
  
  if (!email || !password) {
    throw new Error('Email and password are required');
  }
  
  return {
    id: '1',
    email,
    name: email.split('@')[0],
    role: 'clinician',
    token: 'mock-jwt-token'
  };
};

/**
 * Mock malaria detection API with enhanced location and patient data
 */
const mockDetection = (data) => {
  const { image, location } = data || {};
  
  if (!image) {
    throw new Error('Image is required');
  }
  
  if (!location) {
    throw new Error('Complete location information is required');
  }
  
  // Validate required location fields
  if (!location.province || !location.district || !location.sector || !location.facility) {
    throw new Error('Province, District, Sector, and Healthcare Facility are required');
  }
  
  // Random result with 30% chance of positive detection
  const isPositive = Math.random() < 0.3;
  const confidenceLevel = (Math.random() * 20 + 80).toFixed(1); // 80-100% confidence
  
  // Create comprehensive detection record
  const detection = {
    id: `DX-${Date.now()}-${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
    timestamp: new Date().toISOString(),
    result: isPositive ? 'detected' : 'not_detected',
    confidenceLevel: parseFloat(confidenceLevel),
    imageUrl: image,
    location: {
      province: location.province,
      district: location.district,
      sector: location.sector,
      facility: location.facility,
      coordinates: location.coordinates,
      patientName: location.patientName
    },
    // Additional medical metadata
    metadata: {
      analysisMethod: 'AI-CNN-v2.1',
      processingTime: Math.floor(Math.random() * 3000) + 1000, // 1-4 seconds
      imageQuality: Math.floor(Math.random() * 20) + 80, // 80-100%
      technician: 'AI System',
      reviewRequired: isPositive, // Positive cases require review
    }
  };
  
  // Store detection in localStorage with enhanced structure
  const existingDetections = JSON.parse(localStorage.getItem('malaria_detections') || '[]');
  existingDetections.push(detection);
  
  // Keep only last 1000 detections to prevent storage overflow
  const detectionsToStore = existingDetections.slice(-1000);
  localStorage.setItem('malaria_detections', JSON.stringify(detectionsToStore));
  
  // Also store district-level aggregated data for mapping
  updateDistrictStats(detection);
  
  return detection;
};

/**
 * Update district-level statistics for mapping
 */
const updateDistrictStats = (detection) => {
  const districtKey = `${detection.location.province.id}-${detection.location.district.id}`;
  const existingStats = JSON.parse(localStorage.getItem('district_stats') || '{}');
  
  if (!existingStats[districtKey]) {
    existingStats[districtKey] = {
      districtId: detection.location.district.id,
      districtName: detection.location.district.name,
      provinceName: detection.location.province.name,
      coordinates: detection.location.coordinates,
      totalCases: 0,
      positiveCases: 0,
      facilities: [],
      lastUpdated: new Date().toISOString()
    };
  }
  
  const stats = existingStats[districtKey];
  stats.totalCases += 1;
  if (detection.result === 'detected') {
    stats.positiveCases += 1;
  }
  
  // Ensure facilities is always an array and add facility if not already present
  if (!Array.isArray(stats.facilities)) {
    stats.facilities = [];
  }
  
  if (!stats.facilities.includes(detection.location.facility.name)) {
    stats.facilities.push(detection.location.facility.name);
  }
  
  stats.lastUpdated = new Date().toISOString();
  
  localStorage.setItem('district_stats', JSON.stringify(existingStats));
};

/**
 * Mock statistics API with enhanced district-level data
 */
const mockStatistics = (params) => {
  const { timeRange = 'week' } = params || {};
  
  // Get stored detections and district stats
  const storedDetections = JSON.parse(localStorage.getItem('malaria_detections') || '[]');
  const districtStats = JSON.parse(localStorage.getItem('district_stats') || '{}');
  
  let dataPoints;
  let labels;
  
  switch (timeRange) {
    case 'today':
      labels = Array.from({length: 24}, (_, i) => `${i}:00`);
      dataPoints = 24;
      break;
    case 'week':
      labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
      dataPoints = 7;
      break;
    case 'month':
      labels = Array.from({length: 30}, (_, i) => `Day ${i + 1}`);
      dataPoints = 30;
      break;
    case 'year':
      labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      dataPoints = 12;
      break;
    default:
      labels = Array.from({length: 7}, (_, i) => `Day ${i + 1}`);
      dataPoints = 7;
  }
  
  // Generate data based on actual detections if available, otherwise use mock data
  let positiveData, negativeData;
  
  if (storedDetections.length > 0) {
    // Use actual detection data
    positiveData = Array.from({length: dataPoints}, () => 0);
    negativeData = Array.from({length: dataPoints}, () => 0);
    
    storedDetections.forEach(detection => {
      const randomIndex = Math.floor(Math.random() * dataPoints);
      if (detection.result === 'detected') {
        positiveData[randomIndex] += 1;
      } else {
        negativeData[randomIndex] += 1;
      }
    });
  } else {
    // Use mock data
    positiveData = Array.from({length: dataPoints}, () => Math.floor(Math.random() * 30) + 5);
    negativeData = Array.from({length: dataPoints}, () => Math.floor(Math.random() * 50) + 20);
  }
  
  // Calculate totals
  const totalPositive = positiveData.reduce((sum, val) => sum + val, 0);
  const totalNegative = negativeData.reduce((sum, val) => sum + val, 0);
  const total = totalPositive + totalNegative;
  
  // Generate confidence distribution
  const confidenceBuckets = [
    '0-10%', '11-20%', '21-30%', '31-40%', '41-50%', 
    '51-60%', '61-70%', '71-80%', '81-90%', '91-100%'
  ];
  const confidenceDistribution = confidenceBuckets.map((_, i) => {
    const weight = Math.pow(i + 1, 2);
    return Math.floor(Math.random() * weight * 3) + 1;
  });
  
  return {
    timeRange,
    labels,
    datasets: {
      positive: positiveData,
      negative: negativeData
    },
    summary: {
      total,
      totalPositive,
      totalNegative,
      positiveRate: total > 0 ? ((totalPositive / total) * 100).toFixed(1) : '0.0'
    },
    confidenceDistribution: {
      labels: confidenceBuckets,
      data: confidenceDistribution
    },
    // Enhanced location data with district-level aggregation
    locationData: storedDetections.map(detection => ({
      id: detection.id,
      location: detection.location,
      coordinates: detection.location.coordinates,
      result: detection.result,
      timestamp: detection.timestamp,
      facility: detection.location.facility,
      patientName: detection.location.patientName
    })),
    districtStats: Object.values(districtStats)
  };
};

/**
 * Mock feedback API
 */
const mockFeedback = (data) => {
  const { name, email, type, message } = data || {};
  
  if (!message) {
    throw new Error('Feedback message is required');
  }
  
  return {
    id: Math.floor(Math.random() * 10000),
    timestamp: new Date().toISOString(),
    status: 'received'
  };
};