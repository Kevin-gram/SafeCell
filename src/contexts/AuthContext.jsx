import { createContext, useContext, useState, useEffect } from 'react'
import logger from '../utils/logger'

const AuthContext = createContext(null)

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  
  // Check for existing session on mount
  useEffect(() => {
    const savedUser = localStorage.getItem('user')
    if (savedUser) {
      const userData = JSON.parse(savedUser)
      setUser(userData)
      setIsAuthenticated(true)
      
      // Log session restoration
      logger.logAuth('session_restored', {
        userId: userData.id,
        email: userData.email,
        role: userData.role
      })
    }
    setLoading(false)
  }, [])
  
  // Mock login function
  const login = async (email, password) => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // For demo purposes, accept any non-empty credentials
      if (!email || !password) {
        throw new Error('Email and password are required')
      }
      
      const userData = {
        id: '1',
        email,
        name: email.split('@')[0],
        role: 'clinician',
        location: {
          district: 'Kigali',
          province: 'Kigali City'
        }
      }
      
      setUser(userData)
      setIsAuthenticated(true)
      localStorage.setItem('user', JSON.stringify(userData))
      
      // Log successful login
      logger.logAuth('login_success', {
        userId: userData.id,
        email: userData.email,
        role: userData.role,
        loginMethod: 'email_password'
      })
      
      return { success: true }
    } catch (error) {
      // Log failed login
      logger.logAuth('login_failed', {
        email,
        error: error.message,
        loginMethod: 'email_password'
      })
      
      return { 
        success: false, 
        error: error.message || 'Login failed'
      }
    }
  }
  
  // Update user profile
  const updateUser = async (userData) => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const updatedUser = {
        ...user,
        ...userData
      }
      
      setUser(updatedUser)
      localStorage.setItem('user', JSON.stringify(updatedUser))
      
      // Log profile update
      logger.logAuth('profile_updated', {
        userId: user.id,
        updatedFields: Object.keys(userData)
      })
      
      return { success: true }
    } catch (error) {
      // Log failed profile update
      logger.logAuth('profile_update_failed', {
        userId: user?.id,
        error: error.message
      })
      
      throw new Error('Failed to update profile')
    }
  }
  
  // Logout function
  const logout = () => {
    // Log logout
    logger.logAuth('logout', {
      userId: user?.id,
      sessionDuration: Date.now() - logger.startTime
    })
    
    setUser(null)
    setIsAuthenticated(false)
    localStorage.removeItem('user')
  }
  
  const value = {
    user,
    isAuthenticated,
    loading,
    login,
    logout,
    updateUser
  }
  
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}