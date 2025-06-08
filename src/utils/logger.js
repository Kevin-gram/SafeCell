/**
 * Project Logger for SafeCell
 * Handles logging of user activities, detection results, system events, and performance metrics
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
      // Keep only last 1000 log entries to prevent storage overflow
      const logsToSave = this.logs.slice(-1000)
      localStorage.setItem('safecell_logs', JSON.stringify(logsToSave))
    } catch (error) {
      console.warn('Failed to save logs to localStorage:', error)
    }
  }

  /**
   * Create a base log entry
   */
  createLogEntry(level, category, message, data = {}) {
    return {
      id: `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      sessionId: this.sessionId,
      level,
      category,
      message,
      data,
      userAgent: navigator.userAgent,
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
   * Log malaria detection events
   */
  logDetection(result, data = {}) {
    return this.log('info', 'detection', 'Malaria detection completed', {
      result: result.result,
      confidence: result.confidenceLevel,
      detectionId: result.id,
      processingTime: data.processingTime,
      imageSize: data.imageSize,
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
   * Log user interactions
   */
  logInteraction(action, element, data = {}) {
    return this.log('info', 'interaction', `User ${action} on ${element}`, {
      action,
      element,
      ...data
    })
  }

  /**
   * Log system errors
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
   * Log performance metrics
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
   * Log API requests
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
   * Export logs as JSON
   */
  exportLogs() {
    return JSON.stringify(this.logs, null, 2)
  }

  /**
   * Get log statistics
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
    })

    return stats
  }
}

// Create singleton instance
const logger = new Logger()

// Log initial session start
logger.log('info', 'session', 'Session started', {
  userAgent: navigator.userAgent,
  viewport: {
    width: window.innerWidth,
    height: window.innerHeight
  },
  language: navigator.language,
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
})

export default logger