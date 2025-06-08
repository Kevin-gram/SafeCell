import { motion } from 'framer-motion'
import { FiAlertTriangle, FiCheckCircle, FiInfo, FiXCircle } from 'react-icons/fi'

export default function Alert({ 
  type = 'info',
  title,
  message,
  onClose,
  className = ''
}) {
  const icons = {
    info: <FiInfo className="text-blue-500" size={24} />,
    success: <FiCheckCircle className="text-success-500" size={24} />,
    warning: <FiAlertTriangle className="text-warning-500" size={24} />,
    error: <FiXCircle className="text-error-500" size={24} />
  }
  
  const styles = {
    info: 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-800',
    success: 'bg-success-50 dark:bg-success-900/20 border-success-300 dark:border-success-800',
    warning: 'bg-warning-50 dark:bg-warning-900/20 border-warning-300 dark:border-warning-800',
    error: 'bg-error-50 dark:bg-error-900/20 border-error-300 dark:border-error-800'
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className={`rounded-md border p-4 ${styles[type]} ${className}`}
    >
      <div className="flex items-start">
        <div className="flex-shrink-0">
          {icons[type]}
        </div>
        <div className="ml-3 flex-1">
          {title && <h3 className="text-sm font-medium">{title}</h3>}
          {message && <div className="mt-1 text-sm">{message}</div>}
        </div>
        {onClose && (
          <button
            type="button"
            className="ml-auto flex-shrink-0 text-gray-400 hover:text-gray-500 focus:outline-none"
            onClick={onClose}
          >
            <span className="sr-only">Close</span>
            <FiXCircle className="h-5 w-5" />
          </button>
        )}
      </div>
    </motion.div>
  )
}