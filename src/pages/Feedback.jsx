import { useState } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { useLogger } from '../hooks/useLogger'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import Input from '../components/ui/Input'
import Textarea from '../components/ui/Textarea'
import Select from '../components/ui/Select'
import Alert from '../components/ui/Alert'
import { apiRequest } from '../utils/api'
import { FiSend, FiRefreshCw } from 'react-icons/fi'

export default function Feedback() {
  const { t } = useI18n()
  const { logFeedback, logInteraction, logError } = useLogger()
  
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    type: 'suggestion',
    message: ''
  })
  const [errors, setErrors] = useState({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  
  const feedbackTypes = [
    { value: 'suggestion', label: t('feedback.suggestion') },
    { value: 'bug', label: t('feedback.bug') },
    { value: 'feature', label: t('feedback.feature') },
    { value: 'other', label: t('feedback.other') }
  ]
  
  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    
    // Clear error for this field if any
    if (errors[name]) {
      setErrors(prev => {
        const newErrors = { ...prev }
        delete newErrors[name]
        return newErrors
      })
    }
  }
  
  const validateForm = () => {
    const newErrors = {}
    
    if (!formData.message.trim()) {
      newErrors.message = t('feedback.messageRequired')
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) return
    
    setIsSubmitting(true)
    setErrorMessage('')
    
    logInteraction('submit', 'feedback-form', {
      feedbackType: formData.type,
      hasName: !!formData.name,
      hasEmail: !!formData.email,
      messageLength: formData.message.length
    })
    
    try {
      await apiRequest('/api/feedback', {
        method: 'POST',
        body: formData
      })
      
      // Log successful feedback submission
      logFeedback(formData.type, {
        hasName: !!formData.name,
        hasEmail: !!formData.email,
        messageLength: formData.message.length
      })
      
      setSubmitted(true)
      // Reset form
      setFormData({
        name: '',
        email: '',
        type: 'suggestion',
        message: ''
      })
    } catch (error) {
      const errorMsg = error.message || 'Failed to submit feedback'
      setErrorMessage(errorMsg)
      
      // Log feedback submission error
      logError(error, 'Feedback submission', {
        feedbackType: formData.type,
        messageLength: formData.message.length
      })
    } finally {
      setIsSubmitting(false)
    }
  }
  
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
  
  const resetForm = () => {
    logInteraction('click', 'submit-another-feedback-button')
    setSubmitted(false)
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8 max-w-2xl mx-auto"
    >
      <motion.div variants={itemVariants}>
        <h1 className="text-2xl md:text-3xl font-bold">{t('feedback.title')}</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          {t('feedback.subtitle')}
        </p>
      </motion.div>
      
      <motion.div variants={itemVariants}>
        <Card className="p-6">
          {submitted ? (
            <div className="text-center py-8">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, damping: 10 }}
                className="bg-success-100 dark:bg-success-900/20 p-3 rounded-full inline-flex mb-4"
              >
                <svg className="h-12 w-12 text-success-600 dark:text-success-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </motion.div>
              <h3 className="text-xl font-medium">{t('feedback.thankYou')}</h3>
              <p className="text-gray-600 dark:text-gray-400 mt-2 mb-6">
                Your feedback helps us improve the malaria detection system.
              </p>
              <Button 
                variant="primary" 
                onClick={resetForm}
              >
                Submit Another Feedback
              </Button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              {errorMessage && (
                <Alert 
                  type="error"
                  message={errorMessage}
                  onClose={() => setErrorMessage('')}
                />
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Input
                  label={t('feedback.name')}
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  placeholder="John Doe"
                />
                
                <Input
                  label={t('feedback.email')}
                  name="email"
                  type="email"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="john@example.com"
                />
              </div>
              
              <Select
                label={t('feedback.feedbackType')}
                name="type"
                value={formData.type}
                onChange={handleChange}
                options={feedbackTypes}
                required
              />
              
              <Textarea
                label={t('feedback.message')}
                name="message"
                value={formData.message}
                onChange={handleChange}
                placeholder="Please share your thoughts..."
                required
                error={errors.message}
                rows={6}
              />
              
              <div className="flex justify-end">
                <Button 
                  type="submit" 
                  variant="primary"
                  disabled={isSubmitting}
                  icon={isSubmitting ? <FiRefreshCw className="animate-spin" /> : <FiSend />}
                >
                  {isSubmitting ? t('common.loading') : t('feedback.submit')}
                </Button>
              </div>
            </form>
          )}
        </Card>
      </motion.div>
    </motion.div>
  )
}