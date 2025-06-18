import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { useAuth } from '../contexts/AuthContext'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { FiSearch, FiBarChart2, FiMessageSquare, FiClock, FiActivity } from 'react-icons/fi'

export default function Home() {
  const { t } = useI18n()
  const { user } = useAuth()
  const [timeOfDay, setTimeOfDay] = useState('')
  
  useEffect(() => {
    const hour = new Date().getHours()
    if (hour < 12) setTimeOfDay('morning')
    else if (hour < 17) setTimeOfDay('afternoon')
    else setTimeOfDay('evening')
  }, [])
  
  // Mock recent activity data
  const recentActivity = [
    {
      id: 1,
      action: "Detection completed",
      result: "Positive",
      confidence: 92,
      time: "10 minutes ago"
    },
    {
      id: 2,
      action: "Detection completed",
      result: "Negative",
      confidence: 87,
      time: "1 hour ago"
    },
    {
      id: 3,
      action: "Detection completed",
      result: "Negative",
      confidence: 95,
      time: "3 hours ago"
    }
  ]
  
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.5 }
    }
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      {/* Hero section */}
      <motion.div 
        variants={itemVariants}
        className="bg-gradient-to-r from-primary-600 to-secondary-600 dark:from-primary-800 dark:to-secondary-800 rounded-xl p-8 text-white shadow-lg"
      >
        <h1 className="text-3xl md:text-4xl font-bold">
          Good {timeOfDay}, {user?.name}!
        </h1>
        <p className="mt-2 text-lg text-white/90">
          {t('home.welcome')}
        </p>
        <p className="mt-1 text-white/80 max-w-2xl">
          {t('home.description')}
        </p>
        <div className="mt-6">
          <Link to="/detection">
            <Button variant="accent">
              {t('home.getStarted')}
            </Button>
          </Link>
        </div>
      </motion.div>
      
      {/* Quick access cards */}
      <div>
        <motion.h2 
          variants={itemVariants}
          className="text-xl font-semibold mb-4 flex items-center"
        >
          <FiActivity className="mr-2" /> {t('home.quickAccess')}
        </motion.h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <motion.div variants={itemVariants}>
            <Link to="/detection">
              <Card hoverable className="p-6">
                <div className="flex items-start">
                  <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-full">
                    <FiSearch className="text-primary-600 dark:text-primary-400" size={24} />
                  </div>
                  <div className="ml-4">
                    <h3 className="font-medium">{t('nav.detection')}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Analyze blood smear images
                    </p>
                  </div>
                </div>
              </Card>
            </Link>
          </motion.div>
          
          <motion.div variants={itemVariants}>
            <Link to="/statistics">
              <Card hoverable className="p-6">
                <div className="flex items-start">
                  <div className="p-3 bg-secondary-100 dark:bg-secondary-900/30 rounded-full">
                    <FiBarChart2 className="text-secondary-600 dark:text-secondary-400" size={24} />
                  </div>
                  <div className="ml-4">
                    <h3 className="font-medium">{t('nav.statistics')}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      View detection trends and insights
                    </p>
                  </div>
                </div>
              </Card>
            </Link>
          </motion.div>
          
          <motion.div variants={itemVariants}>
            <Link to="/feedback">
              <Card hoverable className="p-6">
                <div className="flex items-start">
                  <div className="p-3 bg-accent-100 dark:bg-accent-900/30 rounded-full">
                    <FiMessageSquare className="text-accent-600 dark:text-accent-400" size={24} />
                  </div>
                  <div className="ml-4">
                    <h3 className="font-medium">{t('nav.feedback')}</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Provide suggestions for improvement
                    </p>
                  </div>
                </div>
              </Card>
            </Link>
          </motion.div>
        </div>
      </div>
      
      {/* Recent activity */}
      <div>
        <motion.h2 
          variants={itemVariants}
          className="text-xl font-semibold mb-4 flex items-center"
        >
          <FiClock className="mr-2" /> {t('home.recentActivity')}
        </motion.h2>
        
        <motion.div variants={itemVariants}>
          <Card className="overflow-hidden">
            {recentActivity.length > 0 ? (
              <div className="divide-y dark:divide-gray-700">
                {recentActivity.map((activity) => (
                  <div 
                    key={activity.id}
                    className="p-4 flex justify-between items-center hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors"
                  >
                    <div className="flex items-center">
                      <div 
                        className={`h-2 w-2 rounded-full mr-3 ${
                          activity.result === 'Positive' 
                            ? 'bg-error-500' 
                            : 'bg-success-500'
                        }`} 
                      />
                      <div>
                        <p className="font-medium">{activity.action}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {activity.result} • {activity.confidence}% confidence
                        </p>
                      </div>
                    </div>
                    <span className="text-sm text-gray-500 dark:text-gray-400">
                      {activity.time}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-6 text-center text-gray-500 dark:text-gray-400">
                {t('home.noActivity')}
              </div>
            )}
          </Card>
        </motion.div>
      </div>
    </motion.div>
  )
}