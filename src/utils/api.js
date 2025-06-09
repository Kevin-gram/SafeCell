// Simulated API service with logging

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
 * Mock malaria detection API
 */
const mockDetection = (data) => {
  const { image } = data || {};
  
  if (!image) {
    throw new Error('Image is required');
  }
  
  // Random result with 30% chance of positive detection
  const isPositive = Math.random() < 0.3;
  const confidenceLevel = (Math.random() * 20 + 80).toFixed(1); // 80-100% confidence
  
  return {
    id: Math.floor(Math.random() * 10000),
    timestamp: new Date().toISOString(),
    result: isPositive ? 'detected' : 'not_detected',
    confidenceLevel: parseFloat(confidenceLevel),
    imageUrl: image,
  };
};

/**
 * Mock statistics API
 */
const mockStatistics = (params) => {
  const { timeRange = 'week' } = params || {};
  
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
  
  // Generate random data
  const positiveData = Array.from({length: dataPoints}, () => Math.floor(Math.random() * 30) + 5);
  const negativeData = Array.from({length: dataPoints}, () => Math.floor(Math.random() * 50) + 20);
  
  // Calculate totals
  const totalPositive = positiveData.reduce((sum, val) => sum + val, 0);
  const totalNegative = negativeData.reduce((sum, val) => sum + val, 0);
  const total = totalPositive + totalNegative;
  
  // Generate confidence distribution (0-100% in 10% buckets)
  const confidenceBuckets = [
    '0-10%', '11-20%', '21-30%', '31-40%', '41-50%', 
    '51-60%', '61-70%', '71-80%', '81-90%', '91-100%'
  ];
  const confidenceDistribution = confidenceBuckets.map((_, i) => {
    // Higher confidence is more likely
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
      positiveRate: ((totalPositive / total) * 100).toFixed(1)
    },
    confidenceDistribution: {
      labels: confidenceBuckets,
      data: confidenceDistribution
    }
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