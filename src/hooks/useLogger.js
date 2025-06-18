import { useEffect, useRef } from 'react'
import { useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import logger from '../utils/logger'

/**
 * Custom hook for logging user activities and system events
 */
export const useLogger = () => {
  const location = useLocation()
  const { user } = useAuth()
  const previousLocation = useRef(location.pathname)

  // Log navigation changes
  useEffect(() => {
    if (previousLocation.current !== location.pathname) {
      logger.logNavigation(previousLocation.current, location.pathname, {
        userId: user?.id,
        userRole: user?.role
      })
      previousLocation.current = location.pathname
    }
  }, [location.pathname, user])

  // Log page visibility changes
  useEffect(() => {
    const handleVisibilityChange = () => {
      logger.log('info', 'session', `Page ${document.hidden ? 'hidden' : 'visible'}`, {
        userId: user?.id,
        page: location.pathname
      })
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [user, location.pathname])

  // Log window unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      logger.log('info', 'session', 'Session ending', {
        userId: user?.id,
        sessionDuration: Date.now() - logger.startTime
      })
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [user])

  return {
    logAuth: (action, data) => logger.logAuth(action, { userId: user?.id, ...data }),
    logDetection: (result, data) => logger.logDetection(result, { userId: user?.id, ...data }),
    logInteraction: (action, element, data) => logger.logInteraction(action, element, { userId: user?.id, ...data }),
    logError: (error, context, data) => logger.logError(error, context, { userId: user?.id, ...data }),
    logPerformance: (metric, value, data) => logger.logPerformance(metric, value, { userId: user?.id, ...data }),
    logFeedback: (type, data) => logger.logFeedback(type, { userId: user?.id, ...data }),
    logSettings: (setting, oldValue, newValue, data) => logger.logSettings(setting, oldValue, newValue, { userId: user?.id, ...data }),
    logger
  }
}