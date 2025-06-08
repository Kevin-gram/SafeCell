import { useState } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { FiUploadCloud, FiCheckCircle, FiXCircle, FiInfo, FiRefreshCw } from 'react-icons/fi'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { apiRequest } from '../utils/api'

export default function MalariaDetection() {
  const { t } = useI18n()
  
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // Handle image selection
  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    // Reset states
    setResult(null)
    setError(null)
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']
    if (!validTypes.includes(file.type)) {
      setError('Please select a valid image file (JPEG or PNG)')
      return
    }
    
    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      setError('File size exceeds 5MB limit')
      return
    }
    
    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
  }
  
  // Handle image upload and analysis
  const handleAnalyze = async () => {
    if (!selectedImage) return
    
    setIsAnalyzing(true)
    setError(null)
    
    try {
      // Call mock API
      const response = await apiRequest('/api/detection', {
        method: 'POST',
        body: { image: previewUrl }
      })
      
      setResult(response)
    } catch (err) {
      setError(err.message || 'An error occurred during analysis')
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Reset the form
  const handleReset = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
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

  // Sample placeholder images for the detection demo
  const placeholderImages = [
    "https://images.pexels.com/photos/4226119/pexels-photo-4226119.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/4226264/pexels-photo-4226264.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    "https://images.pexels.com/photos/356040/pexels-photo-356040.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
  ]

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-8"
    >
      <motion.div variants={itemVariants}>
        <h1 className="text-2xl md:text-3xl font-bold">{t('detection.title')}</h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          {t('detection.subtitle')}
        </p>
      </motion.div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left column - Upload and Instructions */}
        <motion.div 
          variants={itemVariants}
          className="lg:col-span-1 space-y-6"
        >
          {/* Upload card */}
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4">{previewUrl ? 'Selected Image' : 'Upload Image'}</h3>
            
            {!previewUrl ? (
              <div 
                className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-primary-500 dark:hover:border-primary-400 transition-colors"
                onClick={() => document.getElementById('image-upload').click()}
              >
                <FiUploadCloud size={48} className="mx-auto text-gray-400 dark:text-gray-500" />
                <p className="mt-4 text-gray-600 dark:text-gray-400">{t('detection.dropzone')}</p>
                <input
                  type="file"
                  id="image-upload"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageChange}
                />
                <Button 
                  variant="primary"
                  className="mt-4"
                  icon={<FiUploadCloud />}
                >
                  {t('detection.uploadButton')}
                </Button>
              </div>
            ) : (
              <div>
                <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800">
                  <img 
                    src={previewUrl}
                    alt="Selected blood smear"
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button 
                    variant="primary" 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    icon={isAnalyzing ? <FiRefreshCw className="animate-spin" /> : null}
                  >
                    {isAnalyzing ? t('detection.analyzing') : 'Analyze Image'}
                  </Button>
                  <Button variant="outline" onClick={handleReset}>
                    Reset
                  </Button>
                </div>
                {error && (
                  <div className="mt-3 text-error-600 dark:text-error-400 text-sm">
                    {error}
                  </div>
                )}
              </div>
            )}
          </Card>
          
          {/* Instructions card */}
          <Card className="p-6">
            <h3 className="text-lg font-medium mb-4 flex items-center">
              <FiInfo className="mr-2" /> {t('detection.instructions')}
            </h3>
            <ol className="space-y-3 text-gray-700 dark:text-gray-300 list-decimal pl-5">
              <li>{t('detection.step1')}</li>
              <li>{t('detection.step2')}</li>
              <li>{t('detection.step3')}</li>
            </ol>
          </Card>
        </motion.div>
        
        {/* Right column - Results or Sample Images */}
        <motion.div 
          variants={itemVariants}
          className="lg:col-span-2"
        >
          {result ? (
            // Detection results
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">{t('detection.result')}</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="aspect-video rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800">
                  <img 
                    src={previewUrl}
                    alt="Analyzed blood smear"
                    className="w-full h-full object-cover"
                  />
                </div>
                
                <div className="flex flex-col justify-between">
                  <div>
                    <div className={`rounded-md p-4 ${
                      result.result === 'detected' 
                        ? 'bg-error-100 dark:bg-error-900/20 border border-error-300 dark:border-error-800' 
                        : 'bg-success-100 dark:bg-success-900/20 border border-success-300 dark:border-success-800'
                    }`}>
                      <div className="flex items-center">
                        {result.result === 'detected' ? (
                          <FiXCircle size={24} className="text-error-600 dark:text-error-400" />
                        ) : (
                          <FiCheckCircle size={24} className="text-success-600 dark:text-success-400" />
                        )}
                        <h3 className="ml-2 text-lg font-medium">
                          {result.result === 'detected' 
                            ? t('detection.detected')
                            : t('detection.notDetected')
                          }
                        </h3>
                      </div>
                    </div>
                    
                    <div className="mt-4">
                      <h4 className="text-md font-medium">{t('detection.confidenceLevel')}</h4>
                      <div className="mt-2 relative pt-1">
                        <div className="overflow-hidden h-2 text-xs flex rounded-full bg-gray-200 dark:bg-gray-700">
                          <div 
                            style={{ width: `${result.confidenceLevel}%` }}
                            className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center ${
                              result.result === 'detected'
                                ? 'bg-error-500'
                                : 'bg-success-500'
                            }`}
                          />
                        </div>
                        <div className="mt-1 text-right font-bold">
                          {result.confidenceLevel}%
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4 flex space-x-2">
                    <Button variant="outline" icon={<FiRefreshCw />} onClick={handleReset}>
                      {t('detection.tryAgain')}
                    </Button>
                    <Button variant="primary">
                      {t('detection.viewDetails')}
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            // Sample images
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Sample Blood Smear Images</h2>
              <p className="mb-4 text-gray-600 dark:text-gray-400">
                Upload a blood smear image or select one of these samples to test the malaria detection system.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {placeholderImages.map((img, index) => (
                  <div 
                    key={index}
                    className="aspect-square relative rounded-lg overflow-hidden cursor-pointer group"
                    onClick={() => {
                      setPreviewUrl(img)
                      setSelectedImage({type: 'image/jpeg', size: 1000000}) // mock file object
                    }}
                  >
                    <img 
                      src={img}
                      alt={`Sample ${index + 1}`}
                      className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        <Button variant="primary">Select Sample</Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </motion.div>
      </div>
    </motion.div>
  )
}