import { useState } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { useLogger } from '../hooks/useLogger'
import { FiUploadCloud, FiCheckCircle, FiXCircle, FiInfo, FiRefreshCw, FiMapPin, FiUser, FiHome } from 'react-icons/fi'
import Card from '../components/ui/Card'
import Button from '../components/ui/Button'
import LocationSelector from '../components/detection/LocationSelector'
import { apiRequest } from '../utils/api'
import { getDistrictCoordinates } from '../services/rwandaLocations'

export default function MalariaDetection() {
  const { t } = useI18n()
  const { logDetection, logInteraction, logError, logPerformance } = useLogger()
  
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [selectedLocation, setSelectedLocation] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [locationError, setLocationError] = useState(false)
  
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

  // Handle location selection
  const handleLocationChange = (location) => {
    setSelectedLocation(location)
    setLocationError(false)
  }
  
  // Handle image upload and analysis
  const handleAnalyze = async () => {
    if (!selectedImage) return
    
    // Validate location is selected
    if (!selectedLocation) {
      setLocationError(true)
      setError('Please complete all location fields before analyzing the image')
      return
    }
    
    const startTime = Date.now()
    setIsAnalyzing(true)
    setError(null)
    setLocationError(false)
    
    logInteraction('click', 'analyze-button', {
      fileName: selectedImage.name,
      fileSize: selectedImage.size,
      location: selectedLocation,
      hasPatientName: !!selectedLocation.patientName
    })
    
    try {
      // Get coordinates for the selected district (more accurate than province)
      const coordinates = getDistrictCoordinates(
        selectedLocation.province.id,
        selectedLocation.district.id
      )
      
      // Call detection API with complete location data
      const response = await apiRequest('/api/detection', {
        method: 'POST',
        body: { 
          image: previewUrl,
          location: {
            ...selectedLocation,
            coordinates
          }
        }
      })
      
      const processingTime = Date.now() - startTime
      
      // Log successful detection with complete location data
      logDetection(response, {
        processingTime,
        imageSize: selectedImage.size,
        fileName: selectedImage.name,
        location: selectedLocation,
        coordinates,
        patientName: selectedLocation.patientName,
        facility: selectedLocation.facility
      })
      
      // Log performance metric
      logPerformance('detection_processing_time', processingTime, {
        unit: 'ms',
        imageSize: selectedImage.size,
        location: selectedLocation,
        facility: selectedLocation.facility
      })
      
      setResult(response)
    } catch (err) {
      const errorMsg = err.message || 'An error occurred during analysis'
      setError(errorMsg)
      logError(err, 'Detection analysis', {
        fileName: selectedImage.name,
        fileSize: selectedImage.size,
        processingTime: Date.now() - startTime,
        location: selectedLocation
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
    setLocationError(false)
    // Keep location selected for convenience
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

  // Healthcare-themed blood cell samples
  const bloodCellSamples = [
    {
      url: "/C100P61ThinF_IMG_20150918_144104_cell_128.png",
      title: "Uninfected Blood Cell",
      description: "Normal red blood cell - no malaria parasites detected",
      status: "negative",
      confidence: 94,
      medicalNote: "Healthy erythrocyte with normal morphology"
    },
    {
      url: "/C100P61ThinF_IMG_20150918_144104_cell_163.png", 
      title: "Infected Blood Cell",
      description: "Red blood cell with malaria parasite (Plasmodium)",
      status: "positive",
      confidence: 89,
      medicalNote: "Parasitemia detected - requires immediate treatment"
    },
    {
      url: "https://images.pexels.com/photos/3825527/pexels-photo-3825527.jpeg?auto=compress&cs=tinysrgb&w=800",
      title: "Learn how to do it", 
      description: "Microscopic blood analysis for comparison",
      status: "negative",
      confidence: 92,
      medicalNote: "Normal blood smear examination"
    },
  
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
        <div className="bg-gradient-to-r from-medical-600 to-primary-600 dark:from-medical-800 dark:to-primary-800 rounded-xl p-6 text-white shadow-lg">
          <h1 className="text-2xl md:text-3xl font-bold flex items-center">
            <FiMapPin className="mr-3" />
            Detection Page
          </h1>
          <p className="mt-2 text-medical-100 dark:text-medical-200">
            AI-powered malaria detection system for healthcare professionals
          </p>
          <p className="text-sm text-medical-200 dark:text-medical-300 mt-1">
            Complete location and facility information required for accurate regional health surveillance
          </p>
        </div>
      </motion.div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left column - Location and Upload */}
        <motion.div 
          variants={itemVariants}
          className="lg:col-span-1 space-y-6"
        >
          {/* Location selector */}
          <LocationSelector
            selectedLocation={selectedLocation}
            onLocationChange={handleLocationChange}
            error={locationError}
            required={true}
          />

          {/* Upload card */}
          <Card className="p-6 border-l-4 border-l-medical-500">
            <h3 className="text-lg font-medium mb-4 text-medical-800 dark:text-medical-200 flex items-center">
              <FiUploadCloud className="mr-2" />
              {previewUrl ? 'Selected Image' : 'Upload Blood Smear Image'}
            </h3>
            
            {!previewUrl ? (
              <div 
                className="border-2 border-dashed border-medical-300 dark:border-medical-700 rounded-lg p-8 text-center cursor-pointer hover:border-medical-500 dark:hover:border-medical-400 transition-colors bg-medical-50 dark:bg-medical-900/20"
                onClick={() => {
                  logInteraction('click', 'upload-zone')
                  document.getElementById('image-upload').click()
                }}
              >
                <FiUploadCloud size={48} className="mx-auto text-medical-400 dark:text-medical-500" />
                <p className="mt-4 text-medical-600 dark:text-medical-400 font-medium">
                  {t('detection.dropzone')}
                </p>
                <p className="text-sm text-medical-500 dark:text-medical-500 mt-2">
                  Supported formats: JPEG, PNG (Max 5MB)
                </p>
                <input
                  type="file"
                  id="image-upload"
                  className="hidden"
                  accept="image/*"
                  onChange={handleImageChange}
                />
                <Button 
                  variant="primary"
                  className="mt-4 bg-medical-600 hover:bg-medical-700"
                  icon={<FiUploadCloud />}
                >
                  {t('detection.uploadButton')}
                </Button>
              </div>
            ) : (
              <div>
                <div className="relative aspect-video rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800 border-2 border-medical-200 dark:border-medical-700">
                  <img 
                    src={previewUrl}
                    alt="Selected blood smear"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2 bg-medical-600 text-white px-2 py-1 rounded text-xs font-medium">
                    Ready for Analysis
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button 
                    variant="primary" 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing || !selectedLocation}
                    icon={isAnalyzing ? <FiRefreshCw className="animate-spin" /> : <FiMapPin />}
                    className="bg-medical-600 hover:bg-medical-700"
                  >
                    {isAnalyzing ? t('detection.analyzing') : 'Analyze Blood Smear'}
                  </Button>
                  <Button variant="outline" onClick={handleReset}>
                    Reset
                  </Button>
                </div>
                {error && (
                  <div className="mt-3 text-error-600 dark:text-error-400 text-sm bg-error-50 dark:bg-error-900/20 p-3 rounded-md border border-error-200 dark:border-error-800">
                    {error}
                  </div>
                )}
              </div>
            )}
          </Card>
          
          {/* Instructions card */}
          <Card className="p-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <h3 className="text-lg font-medium mb-4 flex items-center text-blue-800 dark:text-blue-200">
              <FiInfo className="mr-2" /> Clinical Guidelines
            </h3>
            <ol className="space-y-3 text-blue-700 dark:text-blue-300 list-decimal pl-5">
              <li>Complete all location and facility information</li>
              <li>Upload a clear, high-quality blood smear image</li>
              <li>Ensure proper lighting and focus in the microscopic image</li>
              <li>Wait for AI analysis to complete</li>
              <li>Review results and take appropriate clinical action</li>
            </ol>
            
            <div className="mt-4 p-3 bg-blue-100 dark:bg-blue-800/30 rounded-md">
              <p className="text-sm text-blue-800 dark:text-blue-200">
                <strong>Important:</strong> This AI system is designed to assist healthcare professionals in malaria diagnosis. Always confirm results with standard laboratory procedures and clinical judgment.
              </p>
            </div>
          </Card>
        </motion.div>
        
        {/* Right column - Results or Sample Images */}
        <motion.div 
          variants={itemVariants}
          className="lg:col-span-2"
        >
          {result ? (
            // Detection results with complete location and patient info
            <Card className="p-6 border-l-4 border-l-medical-500">
              <h2 className="text-xl font-semibold mb-4 text-medical-800 dark:text-medical-200 flex items-center">
                <FiCheckCircle className="mr-2" />
                Malaria Detection Results
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="aspect-video rounded-lg overflow-hidden bg-gray-100 dark:bg-gray-800 border-2 border-medical-200 dark:border-medical-700">
                  <img 
                    src={previewUrl}
                    alt="Analyzed blood smear"
                    className="w-full h-full object-cover"
                  />
                </div>
                
                <div className="flex flex-col justify-between">
                  <div>
                    {/* Diagnosis Result */}
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
                            ? 'POSITIVE - Malaria Detected'
                            : 'NEGATIVE - No Malaria Detected'
                          }
                        </h3>
                      </div>
                      {result.result === 'detected' && (
                        <p className="mt-2 text-sm text-error-700 dark:text-error-300">
                          <strong>Clinical Action Required:</strong> Malaria parasites detected. Initiate appropriate antimalarial treatment immediately.
                        </p>
                      )}
                    </div>
                    
                    {/* Confidence Level */}
                    <div className="mt-4">
                      <h4 className="text-md font-medium">AI Confidence Level</h4>
                      <div className="mt-2 relative pt-1">
                        <div className="overflow-hidden h-3 text-xs flex rounded-full bg-gray-200 dark:bg-gray-700">
                          <div 
                            style={{ width: `${result.confidenceLevel}%` }}
                            className={`shadow-none flex flex-col text-center whitespace-nowrap text-white justify-center transition-all duration-500 ${
                              result.result === 'detected'
                                ? 'bg-error-500'
                                : 'bg-success-500'
                            }`}
                          />
                        </div>
                        <div className="mt-1 text-right font-bold text-lg">
                          {result.confidenceLevel}%
                        </div>
                      </div>
                    </div>

                    {/* Patient Information */}
                    {result.location?.patientName && (
                      <div className="mt-4 p-3 bg-medical-50 dark:bg-medical-900/20 rounded-md border border-medical-200 dark:border-medical-800">
                        <div className="flex items-center mb-2">
                          <FiUser className="text-medical-500 mr-2" size={16} />
                          <h5 className="text-sm font-medium text-medical-700 dark:text-medical-300">
                            Patient Information
                          </h5>
                        </div>
                        <p className="text-sm text-medical-600 dark:text-medical-400">
                          <strong>Patient:</strong> {result.location.patientName}
                        </p>
                      </div>
                    )}

                    {/* Location and Facility Information */}
                    {result.location && (
                      <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700">
                        <div className="flex items-center mb-2">
                          <FiMapPin className="text-gray-500 mr-2" size={16} />
                          <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Diagnosis Location & Facility
                          </h5>
                        </div>
                        <div className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                          <p><strong>Location:</strong> {result.location.sector.name}, {result.location.district.name}, {result.location.province.name}</p>
                          {result.location.facility && (
                            <p><strong>Healthcare Facility:</strong> {result.location.facility.name}</p>
                          )}
                          <p><strong>Diagnosis ID:</strong> {result.id}</p>
                          <p><strong>Date:</strong> {new Date(result.timestamp).toLocaleString()}</p>
                        </div>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-4 flex space-x-2">
                    <Button variant="outline" icon={<FiRefreshCw />} onClick={handleReset}>
                      New Analysis
                    </Button>
                    <Button 
                      variant="primary"
                      className="bg-medical-600 hover:bg-medical-700"
                      onClick={() => logInteraction('click', 'view-details-button', {
                        detectionId: result.id,
                        result: result.result,
                        location: result.location
                      })}
                    >
                      Save to Records
                    </Button>
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            // Sample blood cell images with healthcare context
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4 text-medical-800 dark:text-medical-200 flex items-center">
                <FiInfo className="mr-2" />
                Clinical Sample Blood Cell Images
              </h2>
              <p className="mb-6 text-gray-600 dark:text-gray-400">
                Upload your own blood smear image or select one of these clinical sample images to test the malaria detection system. 
                These samples represent actual microscopic blood examinations used in malaria diagnosis.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {bloodCellSamples.map((sample, index) => (
                  <div 
                    key={index}
                    className="relative rounded-lg overflow-hidden cursor-pointer group bg-white dark:bg-gray-800 shadow-md hover:shadow-lg transition-all duration-300 border-2 border-gray-200 dark:border-gray-700 hover:border-medical-400"
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
                          <Button variant="primary" size="sm" className="bg-medical-600 hover:bg-medical-700">
                            Analyze Sample
                          </Button>
                        </div>
                      </div>
                      {/* Status indicator */}
                      <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-medium ${
                        sample.status === 'positive' 
                          ? 'bg-error-100 text-error-800 dark:bg-error-900/50 dark:text-error-200'
                          : 'bg-success-100 text-success-800 dark:bg-success-900/50 dark:text-success-200'
                      }`}>
                        {sample.status === 'positive' ? 'Positive' : 'Negative'}
                      </div>
                    </div>
                    <div className="p-4">
                      <h3 className="font-medium text-sm text-medical-800 dark:text-medical-200">{sample.title}</h3>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                        {sample.description}
                      </p>
                      <p className="text-xs text-medical-600 dark:text-medical-400 mt-2 font-medium">
                        {sample.medicalNote}
                      </p>
                      <div className="mt-2 flex items-center justify-between">
                        <span className="text-xs text-gray-500">Expected confidence:</span>
                        <span className="text-xs font-medium text-medical-600 dark:text-medical-400">{sample.confidence}%</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <div className="flex items-start">
                  <FiInfo className="text-amber-600 dark:text-amber-400 mt-0.5 mr-2 flex-shrink-0" />
                  <div>
                    <h4 className="font-medium text-amber-800 dark:text-amber-200">Clinical Sample Information</h4>
                    <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
                      These samples represent actual blood cell images used in clinical malaria diagnosis. The first two samples show the difference between infected and uninfected red blood cells, demonstrating the visual markers our AI system uses for detection.
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