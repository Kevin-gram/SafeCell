import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { useLogger } from '../hooks/useLogger'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Select from '../components/ui/Select'
import Input from '../components/ui/Input'
import { 
  FiFileText, 
  FiDownload, 
  FiTrash2, 
  FiFilter, 
  FiRefreshCw,
  FiActivity,
  FiAlertCircle,
  FiInfo,
  FiUser,
  FiSearch,
  FiBarChart2
} from 'react-icons/fi'

export default function ProjectLogs() {
  const { t } = useI18n()
  const { logger } = useLogger()
  
  const [logs, setLogs] = useState([])
  const [filteredLogs, setFilteredLogs] = useState([])
  const [filters, setFilters] = useState({
    level: 'all',
    category: 'all',
    search: '',
    dateFrom: '',
    dateTo: ''
  })
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  // Load logs and stats
  useEffect(() => {
    const loadData = () => {
      setLoading(true)
      try {
        const allLogs = logger.getAllLogs()
        const logStats = logger.getLogStats()
        
        setLogs(allLogs)
        setStats(logStats)
      } catch (error) {
        console.error('Failed to load logs:', error)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [logger])

  // Apply filters
  useEffect(() => {
    let filtered = [...logs]

    // Filter by level
    if (filters.level !== 'all') {
      filtered = filtered.filter(log => log.level === filters.level)
    }

    // Filter by category
    if (filters.category !== 'all') {
      filtered = filtered.filter(log => log.category === filters.category)
    }

    // Filter by search term
    if (filters.search) {
      const searchTerm = filters.search.toLowerCase()
      filtered = filtered.filter(log => 
        log.message.toLowerCase().includes(searchTerm) ||
        log.category.toLowerCase().includes(searchTerm) ||
        JSON.stringify(log.data).toLowerCase().includes(searchTerm)
      )
    }

    // Filter by date range
    if (filters.dateFrom) {
      const fromDate = new Date(filters.dateFrom).getTime()
      filtered = filtered.filter(log => new Date(log.timestamp).getTime() >= fromDate)
    }

    if (filters.dateTo) {
      const toDate = new Date(filters.dateTo).getTime()
      filtered = filtered.filter(log => new Date(log.timestamp).getTime() <= toDate)
    }

    // Sort by timestamp (newest first)
    filtered.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))

    setFilteredLogs(filtered)
  }, [logs, filters])

  const handleFilterChange = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const clearFilters = () => {
    setFilters({
      level: 'all',
      category: 'all',
      search: '',
      dateFrom: '',
      dateTo: ''
    })
  }

  const exportLogs = () => {
    const dataStr = logger.exportLogs()
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `safecell-logs-${new Date().toISOString().split('T')[0]}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const clearLogs = () => {
    if (window.confirm('Are you sure you want to clear all logs? This action cannot be undone.')) {
      logger.clearLogs()
      setLogs([])
      setFilteredLogs([])
      setStats(logger.getLogStats())
    }
  }

  const refreshLogs = () => {
    const allLogs = logger.getAllLogs()
    const logStats = logger.getLogStats()
    setLogs(allLogs)
    setStats(logStats)
  }

  const getLevelIcon = (level) => {
    switch (level) {
      case 'error':
        return <FiAlertCircle className="text-error-500" />
      case 'warn':
        return <FiAlertCircle className="text-warning-500" />
      case 'info':
        return <FiInfo className="text-blue-500" />
      default:
        return <FiActivity className="text-gray-500" />
    }
  }

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'auth':
        return <FiUser className="text-primary-500" />
      case 'detection':
        return <FiSearch className="text-secondary-500" />
      case 'navigation':
        return <FiActivity className="text-accent-500" />
      default:
        return <FiFileText className="text-gray-500" />
    }
  }

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString()
  }

  const levelOptions = [
    { value: 'all', label: 'All Levels' },
    { value: 'info', label: 'Info' },
    { value: 'warn', label: 'Warning' },
    { value: 'error', label: 'Error' }
  ]

  const categoryOptions = [
    { value: 'all', label: 'All Categories' },
    { value: 'auth', label: 'Authentication' },
    { value: 'detection', label: 'Detection' },
    { value: 'navigation', label: 'Navigation' },
    { value: 'interaction', label: 'User Interaction' },
    { value: 'performance', label: 'Performance' },
    { value: 'api', label: 'API Requests' },
    { value: 'feedback', label: 'Feedback' },
    { value: 'settings', label: 'Settings' },
    { value: 'session', label: 'Session' },
    { value: 'system', label: 'System' }
  ]

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <FiRefreshCw className="animate-spin h-8 w-8 text-primary-600" />
        <span className="ml-2">Loading logs...</span>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center">
            <FiFileText className="mr-2" />
            Project Logs
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            System activity logs and user interactions
          </p>
        </div>
        
        <div className="mt-4 md:mt-0 flex space-x-2">
          <Button
            variant="outline"
            icon={<FiRefreshCw />}
            onClick={refreshLogs}
          >
            Refresh
          </Button>
          <Button
            variant="outline"
            icon={<FiDownload />}
            onClick={exportLogs}
          >
            Export
          </Button>
          <Button
            variant="outline"
            icon={<FiTrash2 />}
            onClick={clearLogs}
          >
            Clear
          </Button>
        </div>
      </div>

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-4">
            <div className="flex items-center">
              <FiBarChart2 className="text-primary-500 mr-2" />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Total Logs</p>
                <p className="text-2xl font-bold">{stats.total}</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center">
              <FiAlertCircle className="text-error-500 mr-2" />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Errors</p>
                <p className="text-2xl font-bold">{stats.byLevel.error || 0}</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center">
              <FiActivity className="text-secondary-500 mr-2" />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Sessions</p>
                <p className="text-2xl font-bold">{stats.sessionCount}</p>
              </div>
            </div>
          </Card>
          
          <Card className="p-4">
            <div className="flex items-center">
              <FiSearch className="text-accent-500 mr-2" />
              <div>
                <p className="text-sm text-gray-600 dark:text-gray-400">Detections</p>
                <p className="text-2xl font-bold">{stats.byCategory.detection || 0}</p>
              </div>
            </div>
          </Card>
        </div>
      )}

      {/* Filters */}
      <Card className="p-6">
        <div className="flex items-center mb-4">
          <FiFilter className="mr-2" />
          <h2 className="text-lg font-semibold">Filters</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <Select
            label="Level"
            value={filters.level}
            onChange={(e) => handleFilterChange('level', e.target.value)}
            options={levelOptions}
          />
          
          <Select
            label="Category"
            value={filters.category}
            onChange={(e) => handleFilterChange('category', e.target.value)}
            options={categoryOptions}
          />
          
          <Input
            label="Search"
            value={filters.search}
            onChange={(e) => handleFilterChange('search', e.target.value)}
            placeholder="Search logs..."
          />
          
          <Input
            label="From Date"
            type="date"
            value={filters.dateFrom}
            onChange={(e) => handleFilterChange('dateFrom', e.target.value)}
          />
          
          <Input
            label="To Date"
            type="date"
            value={filters.dateTo}
            onChange={(e) => handleFilterChange('dateTo', e.target.value)}
          />
        </div>
        
        <div className="mt-4">
          <Button variant="outline" onClick={clearFilters}>
            Clear Filters
          </Button>
        </div>
      </Card>

      {/* Logs Table */}
      <Card className="overflow-hidden">
        <div className="p-4 border-b dark:border-gray-700">
          <h2 className="text-lg font-semibold">
            Log Entries ({filteredLogs.length})
          </h2>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Timestamp
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Level
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Category
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Message
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Data
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {filteredLogs.map((log) => (
                <tr key={log.id} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                  <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                    {formatTimestamp(log.timestamp)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center">
                      {getLevelIcon(log.level)}
                      <span className="ml-2 capitalize">{log.level}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center">
                      {getCategoryIcon(log.category)}
                      <span className="ml-2 capitalize">{log.category}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                    {log.message}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                    {Object.keys(log.data).length > 0 && (
                      <details className="cursor-pointer">
                        <summary className="text-primary-600 dark:text-primary-400 hover:underline">
                          View Data
                        </summary>
                        <pre className="mt-2 text-xs bg-gray-100 dark:bg-gray-800 p-2 rounded overflow-auto max-w-xs">
                          {JSON.stringify(log.data, null, 2)}
                        </pre>
                      </details>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {filteredLogs.length === 0 && (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              No logs found matching the current filters.
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  )
}