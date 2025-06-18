/**
 * Enhanced Project Logger for SafeCell Healthcare System
 * Handles comprehensive logging of healthcare activities, detection results, system events, and performance metrics
 */

class Logger {
  constructor() {
    this.logs = this.loadLogs()
    this.sessionId = this.generateSessionId()
    this.startTime = Date.now()
  }

  /**
   * Generate a unique session ID
   */
  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  /**
   * Load existing logs from localStorage
   */
  loadLogs() {
    try {
      const stored = localStorage.getItem('safecell_logs')
      return stored ? JSON.parse(stored) : []
    } catch (error) {
      console.warn('Failed to load logs from localStorage:', error)
      return []
    }
  }

  /**
   * Save logs to localStorage
   */
  saveLogs() {
    try {
      // Keep only last 2000 log entries to prevent storage overflow
      const logsToSave = this.logs.slice(-2000)
      localStorage.setItem('safecell_logs', JSON.stringify(logsToSave))
    } catch (error) {
      console.warn('Failed to save logs to localStorage:', error)
    }
  }

  /**
   * Create a base log entry with enhanced healthcare context
   */
  createLogEntry(level, category, message, data = {}) {
    return {
      id: `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      sessionId: this.sessionId,
      level,
      category,
      message,
      data: {
        ...data,
        // Add healthcare-specific context
        userAgent: navigator.userAgent,
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight
        },
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        language: navigator.language
      },
      url: window.location.href
    }
  }

  /**
   * Add a log entry
   */
  log(level, category, message, data = {}) {
    const entry = this.createLogEntry(level, category, message, data)
    this.logs.push(entry)
    this.saveLogs()
    
    // Also log to console for development
    if (process.env.NODE_ENV === 'development') {
      console.log(`[${level.toUpperCase()}] ${category}: ${message}`, data)
    }
    
    return entry
  }

  /**
   * Log user authentication events
   */
  logAuth(action, data = {}) {
    return this.log('info', 'auth', `User ${action}`, {
      action,
      ...data
    })
  }

  /**
   * Enhanced malaria detection logging with healthcare context
   */
  logDetection(result, data = {}) {
    return this.log('info', 'detection', 'Malaria detection completed', {
      detectionId: result.id,
      result: result.result,
      confidence: result.confidenceLevel,
      processingTime: data.processingTime,
      imageSize: data.imageSize,
      location: {
        province: data.location?.province?.name,
        district: data.location?.district?.name,
        sector: data.location?.sector?.name,
        facility: data.location?.facility?.name,
        coordinates: data.coordinates
      },
      patient: {
        hasName: !!data.location?.patientName,
        // Don't log actual patient name for privacy
      },
      metadata: {
        timestamp: result.timestamp,
        analysisMethod: result.metadata?.analysisMethod,
        imageQuality: result.metadata?.imageQuality,
        reviewRequired: result.metadata?.reviewRequired
      },
      ...data
    })
  }

  /**
   * Log healthcare facility interactions
   */
  logFacilityInteraction(action, facilityData, data = {}) {
    return this.log('info', 'facility', `Healthcare facility ${action}`, {
      action,
      facility: {
        id: facilityData.id,
        name: facilityData.name,
        type: facilityData.type,
        district: facilityData.district
      },
      ...data
    })
  }

  /**
   * Log patient-related activities (anonymized)
   */
  logPatientActivity(action, data = {}) {
    return this.log('info', 'patient', `Patient ${action}`, {
      action,
      hasPatientName: !!data.patientName,
      // Don't log actual patient data for privacy
      location: data.location,
      timestamp: new Date().toISOString(),
      ...data
    })
  }

  /**
   * Log location selection and validation
   */
  logLocationSelection(locationData, data = {}) {
    return this.log('info', 'location', 'Location selected for diagnosis', {
      location: {
        province: locationData.province?.name,
        district: locationData.district?.name,
        sector: locationData.sector?.name,
        facility: locationData.facility?.name
      },
      coordinates: locationData.coordinates,
      isComplete: !!(locationData.province && locationData.district && locationData.sector && locationData.facility),
      ...data
    })
  }

  /**
   * Log navigation events
   */
  logNavigation(from, to, data = {}) {
    return this.log('info', 'navigation', `Navigated from ${from} to ${to}`, {
      from,
      to,
      ...data
    })
  }

  /**
   * Log user interactions with enhanced healthcare context
   */
  logInteraction(action, element, data = {}) {
    return this.log('info', 'interaction', `User ${action} on ${element}`, {
      action,
      element,
      ...data
    })
  }

  /**
   * Log system errors with healthcare context
   */
  logError(error, context = '', data = {}) {
    return this.log('error', 'system', `Error in ${context}: ${error.message}`, {
      error: {
        name: error.name,
        message: error.message,
        stack: error.stack
      },
      context,
      ...data
    })
  }

  /**
   * Log performance metrics with healthcare-specific metrics
   */
  logPerformance(metric, value, data = {}) {
    return this.log('info', 'performance', `Performance metric: ${metric}`, {
      metric,
      value,
      unit: data.unit || 'ms',
      ...data
    })
  }

  /**
   * Enhanced API request logging
   */
  logApiRequest(endpoint, method, duration, status, data = {}) {
    return this.log('info', 'api', `API ${method} ${endpoint}`, {
      endpoint,
      method,
      duration,
      status,
      ...data
    })
  }

  /**
   * Log feedback submissions
   */
  logFeedback(type, data = {}) {
    return this.log('info', 'feedback', `Feedback submitted: ${type}`, {
      type,
      ...data
    })
  }

  /**
   * Log settings changes
   */
  logSettings(setting, oldValue, newValue, data = {}) {
    return this.log('info', 'settings', `Setting changed: ${setting}`, {
      setting,
      oldValue,
      newValue,
      ...data
    })
  }

  /**
   * Log healthcare compliance and audit events
   */
  logCompliance(event, data = {}) {
    return this.log('info', 'compliance', `Compliance event: ${event}`, {
      event,
      timestamp: new Date().toISOString(),
      ...data
    })
  }

  /**
   * Log data export/import activities
   */
  logDataActivity(activity, data = {}) {
    return this.log('info', 'data', `Data ${activity}`, {
      activity,
      timestamp: new Date().toISOString(),
      ...data
    })
  }

  /**
   * Get logs by category
   */
  getLogsByCategory(category) {
    return this.logs.filter(log => log.category === category)
  }

  /**
   * Get logs by level
   */
  getLogsByLevel(level) {
    return this.logs.filter(log => log.level === level)
  }

  /**
   * Get logs by date range
   */
  getLogsByDateRange(startDate, endDate) {
    const start = new Date(startDate).getTime()
    const end = new Date(endDate).getTime()
    
    return this.logs.filter(log => {
      const logTime = new Date(log.timestamp).getTime()
      return logTime >= start && logTime <= end
    })
  }

  /**
   * Get session logs
   */
  getSessionLogs(sessionId = this.sessionId) {
    return this.logs.filter(log => log.sessionId === sessionId)
  }

  /**
   * Get detection-related logs
   */
  getDetectionLogs() {
    return this.logs.filter(log => log.category === 'detection')
  }

  /**
   * Get facility-related logs
   */
  getFacilityLogs() {
    return this.logs.filter(log => log.category === 'facility')
  }

  /**
   * Get all logs
   */
  getAllLogs() {
    return [...this.logs]
  }

  /**
   * Clear all logs
   */
  clearLogs() {
    this.logs = []
    this.saveLogs()
  }

  /**
   * Export logs as JSON with healthcare-specific formatting
   */
  exportLogs() {
    const exportData = {
      metadata: {
        exportDate: new Date().toISOString(),
        totalLogs: this.logs.length,
        sessionId: this.sessionId,
        system: 'SafeCell Malaria Detection System',
        version: '1.0.0'
      },
      logs: this.logs
    }
    return JSON.stringify(exportData, null, 2)
  }

  /**
   * Get enhanced log statistics with healthcare metrics
   */
  getLogStats() {
    const stats = {
      total: this.logs.length,
      byLevel: {},
      byCategory: {},
      sessionCount: new Set(this.logs.map(log => log.sessionId)).size,
      dateRange: {
        earliest: null,
        latest: null
      },
      healthcare: {
        totalDetections: 0,
        positiveDetections: 0,
        facilitiesUsed: new Set(),
        districtsActive: new Set(),
        patientsProcessed: 0
      }
    }

    this.logs.forEach(log => {
      // Count by level
      stats.byLevel[log.level] = (stats.byLevel[log.level] || 0) + 1
      
      // Count by category
      stats.byCategory[log.category] = (stats.byCategory[log.category] || 0) + 1
      
      // Track date range
      const logDate = new Date(log.timestamp)
      if (!stats.dateRange.earliest || logDate < new Date(stats.dateRange.earliest)) {
        stats.dateRange.earliest = log.timestamp
      }
      if (!stats.dateRange.latest || logDate > new Date(stats.dateRange.latest)) {
        stats.dateRange.latest = log.timestamp
      }

      // Healthcare-specific statistics
      if (log.category === 'detection') {
        stats.healthcare.totalDetections += 1
        if (log.data.result === 'detected') {
          stats.healthcare.positiveDetections += 1
        }
        if (log.data.location?.facility) {
          stats.healthcare.facilitiesUsed.add(log.data.location.facility)
        }
        if (log.data.location?.district) {
          stats.healthcare.districtsActive.add(log.data.location.district)
        }
        if (log.data.patient?.hasName) {
          stats.healthcare.patientsProcessed += 1
        }
      }
    })

    // Convert Sets to counts
    stats.healthcare.facilitiesUsed = stats.healthcare.facilitiesUsed.size
    stats.healthcare.districtsActive = stats.healthcare.districtsActive.size

    return stats
  }
}

// Create singleton instance
const logger = new Logger()

// Log initial session start with healthcare context
logger.log('info', 'session', 'SafeCell healthcare session started', {
  userAgent: navigator.userAgent,
  viewport: {
    width: window.innerWidth,
    height: window.innerHeight
  },
  language: navigator.language,
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  system: 'SafeCell Malaria Detection System',
  version: '1.0.0',
  capabilities: {
    geolocation: 'geolocation' in navigator,
    camera: 'mediaDevices' in navigator,
    localStorage: typeof Storage !== 'undefined'
  }
})

export default logger