import { useEffect } from 'react'
import { Routes, Route, Navigate, useLocation } from 'react-router-dom'
import { useAuth } from './contexts/AuthContext'
import { useLogger } from './hooks/useLogger'

// Layouts
import Layout from './components/layout/Layout'

// Pages
import Login from './pages/Login'
import Home from './pages/Home'
import MalariaDetection from './pages/MalariaDetection'
import Statistics from './pages/Statistics'
import LocationStats from './pages/LocationStats'
import Settings from './pages/Settings'
import Feedback from './pages/Feedback'
import NotFound from './pages/NotFound'

const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuth()
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  return children
}

export default function App() {
  const { isAuthenticated } = useAuth()
  const location = useLocation()
  const { logError } = useLogger()

  // Scroll to top on route change
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location.pathname])

  // Global error handler
  useEffect(() => {
    const handleError = (event) => {
      logError(event.error, 'Global error handler', {
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno
      })
    }

    const handleUnhandledRejection = (event) => {
      logError(new Error(event.reason), 'Unhandled promise rejection')
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleUnhandledRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleUnhandledRejection)
    }
  }, [logError])

  return (
    <Routes>
      <Route 
        path="/login" 
        element={isAuthenticated ? <Navigate to="/" replace /> : <Login />} 
      />
      
      <Route path="/" element={
        <ProtectedRoute>
          <Layout />
        </ProtectedRoute>
      }>
        <Route index element={<Home />} />
        <Route path="detection" element={<MalariaDetection />} />
        <Route path="statistics" element={<Statistics />} />
        <Route path="location-stats" element={<LocationStats />} />
        <Route path="settings" element={<Settings />} />
        <Route path="feedback" element={<Feedback />} />
      </Route>
      
      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}