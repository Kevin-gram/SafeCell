import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { apiRequest } from '../utils/api'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  ArcElement 
} from 'chart.js'
import { Line, Bar, Doughnut } from 'react-chartjs-2'
import { FiBarChart2, FiPieChart, FiRefreshCw, FiTrendingUp } from 'react-icons/fi'

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
)

export default function Statistics() {
  const { t } = useI18n()
  
  const [timeRange, setTimeRange] = useState('week')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [stats, setStats] = useState(null)
  
  const fetchStatistics = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const data = await apiRequest('/api/statistics', {
        params: { timeRange }
      })
      setStats(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }
  
  useEffect(() => {
    fetchStatistics()
  }, [timeRange])
  
  const timeRangeOptions = [
    { value: 'today', label: t('statistics.today') },
    { value: 'week', label: t('statistics.week') },
    { value: 'month', label: t('statistics.month') },
    { value: 'year', label: t('statistics.year') }
  ]
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  }
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.4 } }
  }
  
  const getChartColors = (isDarkMode = false) => {
    return {
      positive: {
        backgroundColor: isDarkMode ? 'rgba(239, 68, 68, 0.2)' : 'rgba(239, 68, 68, 0.2)',
        borderColor: isDarkMode ? 'rgb(239, 68, 68)' : 'rgb(239, 68, 68)',
      },
      negative: {
        backgroundColor: isDarkMode ? 'rgba(16, 185, 129, 0.2)' : 'rgba(16, 185, 129, 0.2)',
        borderColor: isDarkMode ? 'rgb(16, 185, 129)' : 'rgb(16, 185, 129)',
      }
    }
  }
  
  const getDetectionRateData = () => {
    if (!stats) return null
    
    const colors = getChartColors(document.documentElement.classList.contains('dark'))
    
    return {
      labels: stats.labels,
      datasets: [
        {
          label: t('statistics.positiveDetections'),
          data: stats.datasets.positive,
          backgroundColor: colors.positive.backgroundColor,
          borderColor: colors.positive.borderColor,
          borderWidth: 2,
          tension: 0.3
        },
        {
          label: t('statistics.negativeDetections'),
          data: stats.datasets.negative,
          backgroundColor: colors.negative.backgroundColor,
          borderColor: colors.negative.borderColor,
          borderWidth: 2,
          tension: 0.3
        }
      ]
    }
  }
  
  const getConfidenceDistributionData = () => {
    if (!stats?.confidenceDistribution) return null
    
    return {
      labels: stats.confidenceDistribution.labels,
      datasets: [
        {
          label: 'Count',
          data: stats.confidenceDistribution.data,
          backgroundColor: [
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)',
            'rgba(255, 159, 64, 0.6)',
            'rgba(255, 99, 132, 0.6)',
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)',
            'rgba(153, 102, 255, 1)',
            'rgba(255, 159, 64, 1)',
            'rgba(255, 99, 132, 1)',
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(75, 192, 192, 1)'
          ],
          borderWidth: 1
        }
      ]
    }
  }
  
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      <motion.div variants={itemVariants} className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold">{t('statistics.title')}</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            {t('statistics.subtitle')}
          </p>
        </div>
        
        <div className="mt-4 md:mt-0">
          <div className="inline-flex rounded-md shadow-sm">
            {timeRangeOptions.map((option) => (
              <button
                key={option.value}
                onClick={() => setTimeRange(option.value)}
                className={`
                  px-4 py-2 text-sm font-medium
                  ${timeRange === option.value 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }
                  ${option.value === 'today' ? 'rounded-l-md' : ''}
                  ${option.value === 'year' ? 'rounded-r-md' : ''}
                  border border-gray-300 dark:border-gray-600
                  focus:z-10 focus:outline-none focus:ring-1 focus:ring-primary-500
                `}
              >
                {option.label}
              </button>
            ))}
          </div>
          <Button 
            variant="outline" 
            size="sm"
            icon={<FiRefreshCw className={loading ? 'animate-spin' : ''} />}
            onClick={fetchStatistics}
            className="ml-2"
            disabled={loading}
          >
            Refresh
          </Button>
        </div>
      </motion.div>
      
      {error ? (
        <motion.div variants={itemVariants} className="p-4 bg-error-100 dark:bg-error-900/20 text-error-700 dark:text-error-400 rounded-md">
          {error}
        </motion.div>
      ) : loading ? (
        <motion.div variants={itemVariants} className="flex justify-center p-12">
          <div className="animate-pulse text-center">
            <FiRefreshCw className="animate-spin mx-auto h-8 w-8 text-primary-600 dark:text-primary-400" />
            <p className="mt-2 text-gray-600 dark:text-gray-400">{t('common.loading')}</p>
          </div>
        </motion.div>
      ) : stats ? (
        <>
          {/* Summary stats */}
          <motion.div variants={itemVariants} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card className="p-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Scans</p>
              <p className="mt-1 text-2xl font-semibold">{stats.summary.total}</p>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Positive Cases</p>
              <p className="mt-1 text-2xl font-semibold text-error-600 dark:text-error-400">{stats.summary.totalPositive}</p>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Negative Cases</p>
              <p className="mt-1 text-2xl font-semibold text-success-600 dark:text-success-400">{stats.summary.totalNegative}</p>
            </Card>
            <Card className="p-4">
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Positive Rate</p>
              <p className="mt-1 text-2xl font-semibold">{stats.summary.positiveRate}%</p>
            </Card>
          </motion.div>
          
          {/* Detection rate chart */}
          <motion.div variants={itemVariants}>
            <Card className="p-6">
              <div className="flex items-center mb-4">
                <FiTrendingUp size={20} className="text-primary-600 dark:text-primary-400 mr-2" />
                <h2 className="text-xl font-semibold">{t('statistics.detectionRate')}</h2>
              </div>
              <div className="h-80">
                <Line 
                  data={getDetectionRateData()} 
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'top',
                      },
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        grid: {
                          color: document.documentElement.classList.contains('dark') 
                            ? 'rgba(255, 255, 255, 0.1)' 
                            : 'rgba(0, 0, 0, 0.1)'
                        }
                      },
                      x: {
                        grid: {
                          color: document.documentElement.classList.contains('dark') 
                            ? 'rgba(255, 255, 255, 0.05)' 
                            : 'rgba(0, 0, 0, 0.05)'
                        }
                      }
                    }
                  }}
                />
              </div>
            </Card>
          </motion.div>
          
          {/* Additional charts */}
          <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Positive/Negative distribution */}
            <Card className="p-6">
              <div className="flex items-center mb-4">
                <FiBarChart2 size={20} className="text-secondary-600 dark:text-secondary-400 mr-2" />
                <h2 className="text-xl font-semibold">Positive/Negative Distribution</h2>
              </div>
              <div className="h-64">
                <Bar 
                  data={{
                    labels: ['Positive', 'Negative'],
                    datasets: [
                      {
                        label: 'Count',
                        data: [stats.summary.totalPositive, stats.summary.totalNegative],
                        backgroundColor: [
                          'rgba(239, 68, 68, 0.6)',
                          'rgba(16, 185, 129, 0.6)'
                        ],
                        borderColor: [
                          'rgb(239, 68, 68)',
                          'rgb(16, 185, 129)'
                        ],
                        borderWidth: 1
                      }
                    ]
                  }}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        display: false
                      }
                    }
                  }}
                />
              </div>
            </Card>
            
            {/* Confidence distribution */}
            <Card className="p-6">
              <div className="flex items-center mb-4">
                <FiPieChart size={20} className="text-accent-600 dark:text-accent-400 mr-2" />
                <h2 className="text-xl font-semibold">{t('statistics.confidenceDistribution')}</h2>
              </div>
              <div className="h-64">
                <Doughnut
                  data={getConfidenceDistributionData()}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'right'
                      }
                    }
                  }}
                />
              </div>
            </Card>
          </motion.div>
        </>
      ) : (
        <motion.div variants={itemVariants} className="text-center p-12">
          <p>{t('statistics.noData')}</p>
        </motion.div>
      )}
    </motion.div>
  )
}