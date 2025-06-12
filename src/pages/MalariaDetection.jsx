import { useState } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { useLogger } from '../hooks/useLogger'
import { FiUploadCloud, FiCheckCircle, FiXCircle, FiInfo, FiRefreshCw } from 'react-icons/fi'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import { apiRequest } from '../utils/api'

export default function MalariaDetection() {
  const { t } = useI18n()
  const { logDetection, logInteraction, logError, logPerformance } = useLogger()
  
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  
  // Handle image selection
  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    // Log interaction
    logInteraction('upload', 'image-file', {
      fileName: file.name,
      fileSize: file.size,
      fileType: file.type
    })
    
    // Reset states
    setResult(null)
    setError(null)
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg']
    if (!validTypes.includes(file.type)) {
      const errorMsg = 'Please select a valid image file (JPEG or PNG)'
      setError(errorMsg)
      logError(new Error(errorMsg), 'Image validation', {
        fileName: file.name,
        fileType: file.type
      })
      return
    }
    
    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      const errorMsg = 'File size exceeds 5MB limit'
      setError(errorMsg)
      logError(new Error(errorMsg), 'Image validation', {
        fileName: file.name,
        fileSize: file.size
      })
      return
    }
    
    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
  }
  
  // Handle image upload and analysis
  const handleAnalyze = async () => {
    if (!selectedImage) return
    
    const startTime = Date.now()
    setIsAnalyzing(true)
    setError(null)
    
    logInteraction('click', 'analyze-button', {
      fileName: selectedImage.name,
      fileSize: selectedImage.size
    })
    
    try {
      // Call mock API
      const response = await apiRequest('/api/detection', {
        method: 'POST',
        body: { image: previewUrl }
      })
      
      const processingTime = Date.now() - startTime
      
      // Log successful detection
      logDetection(response, {
        processingTime,
        imageSize: selectedImage.size,
        fileName: selectedImage.name
      })
      
      // Log performance metric
      logPerformance('detection_processing_time', processingTime, {
        unit: 'ms',
        imageSize: selectedImage.size
      })
      
      setResult(response)
    } catch (err) {
      const errorMsg = err.message || 'An error occurred during analysis'
      setError(errorMsg)
      logError(err, 'Detection analysis', {
        fileName: selectedImage.name,
        fileSize: selectedImage.size,
        processingTime: Date.now() - startTime
      })
    } finally {
      setIsAnalyzing(false)
    }
  }
  
  // Reset the form
  const handleReset = () => {
    logInteraction('click', 'reset-button')
    
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

  // Actual blood cell images for malaria detection
  const bloodCellSamples = [
    {
      url: "/C100P61ThinF_IMG_20150918_144104_cell_128.png",
      title: "Uninfected Blood Cell",
      description: "Normal red blood cell - no malaria parasites detected",
      status: "negative",
      confidence: 94
    },
    {
      url: "/C100P61ThinF_IMG_20150918_144104_cell_163.png", 
      title: "Infected Blood Cell",
      description: "Red blood cell with malaria parasite (Plasmodium)",
      status: "positive",
      confidence: 89
    },
    {
      url: "https://images.pexels.com/photos/3825527/pexels-photo-3825527.jpeg?auto=compress&cs=tinysrgb&w=800",
      title: "Blood Smear Sample 3", 
      description: "Microscopic blood analysis for comparison",
      status: "negative",
      confidence: 92
    },
    {
      url: "https://images.pexels.com/photos/3825539/pexels-photo-3825539.jpeg?auto=compress&cs=tinysrgb&w=800",
      title: "Blood Smear Sample 4",
      description: "Laboratory blood examination sample",
      status: "negative", 
      confidence: 87
    }
  ]

  const handleSampleSelect = (sample, index) => {
    logInteraction('click', 'sample-image', {
      sampleIndex: index,
      imageUrl: sample.url,
      sampleTitle: sample.title,
      expectedStatus: sample.status
    })
    
    setPreviewUrl(sample.url)
    setSelectedImage({
      type: 'image/png', 
      size: 500000, 
      name: `blood-cell-sample-${index + 1}.png`
    }) // mock file object
    setResult(null)
    setError(null)
  }

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
                onClick={() => {
                  logInteraction('click', 'upload-zone')
                  document.getElementById('image-upload').click()
                }}
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
            
            <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>Note:</strong> For best results, ensure blood smear images are:
              </p>
              <ul className="text-sm text-blue-700 dark:text-blue-300 mt-2 list-disc pl-4">
                <li>High resolution and well-focused</li>
                <li>Properly stained (Giemsa or similar)</li>
                <li>Free from artifacts or debris</li>
                <li>Captured under appropriate magnification (1000x recommended)</li>
              </ul>
            </div>
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
                      {result.result === 'detected' && (
                        <p className="mt-2 text-sm text-error-700 dark:text-error-300">
                          Malaria parasites detected in blood smear. Immediate medical attention recommended.
                        </p>
                      )}
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
                    <Button 
                      variant="primary"
                      onClick={() => logInteraction('click', 'view-details-button', {
                        detectionId: result.id,
                        result: result.result
                      })}
                    >
                      {t('detection.viewDetails')}
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            // Sample blood cell images
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Sample Blood Cell Images</h2>
              <p className="mb-6 text-gray-600 dark:text-gray-400">
                Upload your own blood smear image or select one of these sample blood cell images to test the malaria detection system. 
                These samples include both infected and uninfected cells for demonstration.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {bloodCellSamples.map((sample, index) => (
                  <div 
                    key={index}
                    className="relative rounded-lg overflow-hidden cursor-pointer group bg-white dark:bg-gray-800 shadow-md hover:shadow-lg transition-all duration-300 border dark:border-gray-700"
                    onClick={() => handleSampleSelect(sample, index)}
                  >
                    <div className="aspect-square relative">
                      <img 
                        src={sample.url}
                        alt={sample.title}
                        className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all duration-300 flex items-center justify-center">
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                          <Button variant="primary" size="sm">
                            Select Sample
                          </Button>
                        </div>
                      </div>
                      {/* Status indicator */}
                      <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium ${
                        sample.status === 'positive' 
                          ? 'bg-error-100 text-error-800 dark:bg-error-900/50 dark:text-error-200'
                          : 'bg-success-100 text-success-800 dark:bg-success-900/50 dark:text-success-200'
                      }`}>
                        {sample.status === 'positive' ? 'Infected' : 'Uninfected'}
                      </div>
                    </div>
                    <div className="p-4">
                      <h3 className="font-medium text-sm">{sample.title}</h3>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        {sample.description}
                      </p>
                      <div className="mt-2 flex items-center justify-between">
                        <span className="text-xs text-gray-500">Expected confidence:</span>
                        <span className="text-xs font-medium">{sample.confidence}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <div className="flex items-start">
                  <FiInfo className="text-amber-600 dark:text-amber-400 mt-0.5 mr-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-amber-800 dark:text-amber-200">Sample Images Information</h4>
                    <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                      The first two samples are actual blood cell images: one uninfected and one infected with malaria parasites. 
                      These demonstrate the visual differences the AI system looks for when detecting malaria.
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          )}
        </motion.div>
      </div>
    </motion.div>
  )
}