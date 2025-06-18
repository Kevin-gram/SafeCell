import { Link } from 'react-router-dom'
import { useI18n } from '../contexts/I18nContext'
import { motion } from 'framer-motion'
import Button from '../components/ui/Button'

export default function NotFound() {
  const { t } = useI18n()
  
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 flex items-center justify-center px-4 sm:px-6 lg:px-8">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <h1 className="text-6xl font-bold text-primary-600 dark:text-primary-400">404</h1>
        <h2 className="mt-4 text-3xl font-bold text-gray-900 dark:text-white">
          {t('notFound.title')}
        </h2>
        <p className="mt-2 text-lg text-gray-600 dark:text-gray-400">
          {t('notFound.message')}
        </p>
        <div className="mt-6">
          <Link to="/">
            <Button variant="primary">
              {t('notFound.returnHome')}
            </Button>
          </Link>
        </div>
      </motion.div>
    </div>
  )
}