import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { MapContainer, TileLayer, Circle, Popup, Marker } from 'react-leaflet'
import Card from '../components/ui/Card'
import Select from '../components/ui/Select'
import { 
  FiMapPin, 
  FiAlertTriangle, 
  FiActivity, 
  FiTrendingUp, 
  FiHome,
  FiUsers,
  FiBarChart2
} from 'react-icons/fi'
import { apiRequest } from '../utils/api'
import 'leaflet/dist/leaflet.css'

// Rwanda's center coordinates and bounds
const RWANDA_CENTER = [-1.9403, 29.8739]
const RWANDA_BOUNDS = [
  [-2.8389, 28.8617], // Southwest
  [-1.0474, 30.8862]  // Northeast
]

// Enhanced intensity levels for healthcare context
const INTENSITY_LEVELS = [
  { level: 1, name: 'Low Risk', color: '#10B981', description: 'Minimal cases (1-5 per district)', priority: 'routine' },
  { level: 2, name: 'Moderate Risk', color: '#FBBF24', description: 'Some cases detected (6-15 per district)', priority: 'monitor' },
  { level: 3, name: 'High Risk', color: '#F59E0B', description: 'Significant cases (16-30 per district)', priority: 'alert' },
  { level: 4, name: 'Very High Risk', color: '#DC2626', description: 'Heavy case load (31-50 per district)', priority: 'urgent' },
  { level: 5, name: 'Critical Risk', color: '#991B1B', description: 'Outbreak level (>50 per district)', priority: 'emergency' }
]

export default function LocationStats() {
  const { t } = useI18n()
  const [timeRange, setTimeRange] = useState('month')
  const [loading, setLoading] = useState(true)
  const [districtData, setDistrictData] = useState([])
  const [detectionData, setDetectionData] = useState([])
  const [stats, setStats] = useState(null)
  
  // Fetch district-level data and detection data
  useEffect(() => {
    const fetchLocationData = async () => {
      setLoading(true)
      try {
        // Get statistics data which includes district-level data
        const statisticsData = await apiRequest('/api/statistics', {
          params: { timeRange }
        })
        
        // Get stored detections and district stats from localStorage
        const storedDetections = JSON.parse(localStorage.getItem('malaria_detections') || '[]')
        const districtStats = JSON.parse(localStorage.getItem('district_stats') || '{}')
        
        // Process district-level data for mapping
        const processedDistrictData = Object.values(districtStats).map(district => ({
          id: district.districtId,
          name: district.districtName,
          province: district.provinceName,
          coordinates: [district.coordinates.lat, district.coordinates.lng],
          totalCases: district.totalCases,
          positiveCases: district.positiveCases,
          facilities: district.facilities,
          intensity: calculateIntensity(district.totalCases),
          positiveRate: district.totalCases > 0 ? ((district.positiveCases / district.totalCases) * 100).toFixed(1) : 0,
          lastUpdated: district.lastUpdated,
          riskLevel: getRiskLevel(district.totalCases, district.positiveCases)
        }))
        
        // Add mock data if no real detections exist
        if (processedDistrictData.length === 0) {
          const mockDistrictData = [
            {
              id: 'gasabo',
              name: 'Gasabo',
              province: 'Kigali City',
              coordinates: [-1.9403, 30.0615],
              totalCases: 45,
              positiveCases: 12,
              facilities: ['Kigali University Teaching Hospital', 'King Faisal Hospital'],
              intensity: 3,
              positiveRate: 26.7,
              riskLevel: 'moderate',
              lastUpdated: new Date().toISOString()
            },
            {
              id: 'musanze',
              name: 'Musanze',
              province: 'Northern Province',
              coordinates: [-1.4995, 29.6335],
              totalCases: 23,
              positiveCases: 8,
              facilities: ['Ruhengeri Referral Hospital'],
              intensity: 2,
              positiveRate: 34.8,
              riskLevel: 'low',
              lastUpdated: new Date().toISOString()
            },
            {
              id: 'huye',
              name: 'Huye',
              province: 'Southern Province',
              coordinates: [-2.6399, 29.7406],
              totalCases: 67,
              positiveCases: 19,
              facilities: ['Butare University Teaching Hospital'],
              intensity: 4,
              positiveRate: 28.4,
              riskLevel: 'high',
              lastUpdated: new Date().toISOString()
            }
          ]
          setDistrictData(mockDistrictData)
        } else {
          setDistrictData(processedDistrictData)
        }
        
        setDetectionData(storedDetections)
        setStats(statisticsData)
        
      } catch (error) {
        console.error('Failed to fetch location data:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchLocationData()
  }, [timeRange])
  
  const calculateIntensity = (cases) => {
    if (cases <= 5) return 1
    if (cases <= 15) return 2
    if (cases <= 30) return 3
    if (cases <= 50) return 4
    return 5
  }
  
  const getRiskLevel = (totalCases, positiveCases) => {
    const positiveRate = totalCases > 0 ? (positiveCases / totalCases) * 100 : 0
    if (positiveRate > 40) return 'critical'
    if (positiveRate > 30) return 'high'
    if (positiveRate > 20) return 'moderate'
    return 'low'
  }
  
  const getCircleColor = (intensity) => {
    return INTENSITY_LEVELS[intensity - 1]?.color || '#000000'
  }
  
  const getCircleSize = (cases) => {
    // Scale circle size based on number of cases (district level)
    return Math.sqrt(cases) * 2000 // Larger circles for district-level data
  }
  
  const timeRangeOptions = [
    { value: 'today', label: 'Today' },
    { value: 'week', label: 'This Week' },
    { value: 'month', label: 'This Month' },
    { value: 'year', label: 'This Year' }
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8"
    >
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl md:text-3xl font-bold flex items-center text-medical-800 dark:text-medical-200">
            <FiMapPin className="mr-2 text-medical-600" />
            District-Level Malaria Surveillance
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Real-time malaria distribution and intensity mapping across Rwanda's districts based on healthcare facility diagnoses
          </p>
        </div>
        
        <div className="mt-4 md:mt-0">
          <Select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            options={timeRangeOptions}
            className="w-48"
          />
        </div>
      </div>

      {/* Enhanced Summary Statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="p-4 border-l-4 border-l-medical-500">
          <div className="flex items-center">
            <FiMapPin className="text-medical-500 mr-2" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Active Districts</p>
              <p className="text-2xl font-bold text-medical-600">{districtData.length}</p>
            </div>
          </div>
        </Card>
        
        <Card className="p-4 border-l-4 border-l-blue-500">
          <div className="flex items-center">
            <FiActivity className="text-blue-500 mr-2" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Total Diagnoses</p>
              <p className="text-2xl font-bold text-blue-600">
                {districtData.reduce((sum, district) => sum + district.totalCases, 0)}
              </p>
            </div>
          </div>
        </Card>
        
        <Card className="p-4 border-l-4 border-l-error-500">
          <div className="flex items-center">
            <FiAlertTriangle className="text-error-500 mr-2" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Positive Cases</p>
              <p className="text-2xl font-bold text-error-600">
                {districtData.reduce((sum, district) => sum + district.positiveCases, 0)}
              </p>
            </div>
          </div>
        </Card>
        
        <Card className="p-4 border-l-4 border-l-success-500">
          <div className="flex items-center">
            <FiHome className="text-success-500 mr-2" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Healthcare Facilities</p>
              <p className="text-2xl font-bold text-success-600">
                {new Set(districtData.flatMap(d => d.facilities)).size}
              </p>
            </div>
          </div>
        </Card>
        
        <Card className="p-4 border-l-4 border-l-accent-500">
          <div className="flex items-center">
            <FiTrendingUp className="text-accent-500 mr-2" />
            <div>
              <p className="text-sm text-gray-600 dark:text-gray-400">Avg Positive Rate</p>
              <p className="text-2xl font-bold text-accent-600">
                {districtData.length > 0 
                  ? (districtData.reduce((sum, district) => sum + parseFloat(district.positiveRate), 0) / districtData.length).toFixed(1)
                  : 0
                }%
              </p>
            </div>
          </div>
        </Card>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Enhanced Risk Levels Card */}
        <Card className="p-6 border-l-4 border-l-medical-500">
          <h2 className="text-xl font-semibold mb-4 flex items-center text-medical-800 dark:text-medical-200">
            <FiAlertTriangle className="mr-2" />
            Risk Assessment Levels
          </h2>
          <div className="space-y-4">
            {INTENSITY_LEVELS.map(({ level, name, color, description, priority }) => (
              <div key={level} className="flex items-start space-x-3">
                <div 
                  className="w-4 h-4 rounded-full mt-1 flex-shrink-0"
                  style={{ backgroundColor: color }}
                />
                <div>
                  <h3 className="font-medium text-sm">Level {level} - {name}</h3>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {description}
                  </p>
                  <span className={`inline-block mt-1 px-2 py-1 text-xs rounded-full ${
                    priority === 'emergency' ? 'bg-error-100 text-error-800 dark:bg-error-900/50 dark:text-error-200' :
                    priority === 'urgent' ? 'bg-warning-100 text-warning-800 dark:bg-warning-900/50 dark:text-warning-200' :
                    priority === 'alert' ? 'bg-accent-100 text-accent-800 dark:bg-accent-900/50 dark:text-accent-200' :
                    priority === 'monitor' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200' :
                    'bg-success-100 text-success-800 dark:bg-success-900/50 dark:text-success-200'
                  }`}>
                    {priority.toUpperCase()}
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          {detectionData.length > 0 && (
            <div className="mt-6 p-3 bg-medical-50 dark:bg-medical-900/20 rounded-md border border-medical-200 dark:border-medical-800">
              <p className="text-sm text-medical-800 dark:text-medical-200">
                <strong>Live Data:</strong> Based on {detectionData.length} actual diagnoses from {new Set(districtData.flatMap(d => d.facilities)).size} healthcare facilities across Rwanda.
              </p>
            </div>
          )}
        </Card>
        
        {/* Enhanced Map Card */}
        <Card className="lg:col-span-2 overflow-hidden border-l-4 border-l-medical-500">
          <div className="p-4 border-b dark:border-gray-700 bg-medical-50 dark:bg-medical-900/20">
            <h2 className="text-xl font-semibold text-medical-800 dark:text-medical-200">
              District-Level Malaria Distribution Map
            </h2>
            <p className="text-sm text-medical-600 dark:text-medical-400 mt-1">
              Circle size represents total cases per district, color indicates risk intensity level
            </p>
          </div>
          <div className="h-[600px]">
            {loading ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-medical-600 mx-auto"></div>
                  <p className="mt-2 text-gray-600 dark:text-gray-400">Loading district data...</p>
                </div>
              </div>
            ) : (
              <MapContainer
                center={RWANDA_CENTER}
                zoom={8}
                maxBounds={RWANDA_BOUNDS}
                className="h-full w-full"
              >
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />
                
                {/* District-level circles */}
                {districtData.map((district, index) => (
                  <Circle
                    key={`district-${index}`}
                    center={district.coordinates}
                    radius={getCircleSize(district.totalCases)}
                    pathOptions={{
                      color: getCircleColor(district.intensity),
                      fillColor: getCircleColor(district.intensity),
                      fillOpacity: 0.6,
                      weight: 3
                    }}
                  >
                    <Popup>
                      <div className="p-3 min-w-[250px]">
                        <h3 className="font-bold text-lg text-medical-800">{district.name} District</h3>
                        <p className="text-sm text-gray-600 mb-2">{district.province}</p>
                        <div className="space-y-1 text-sm">
                          <p><strong>Total Cases:</strong> {district.totalCases}</p>
                          <p><strong>Positive Cases:</strong> {district.positiveCases}</p>
                          <p><strong>Positive Rate:</strong> {district.positiveRate}%</p>
                          <p><strong>Risk Level:</strong> 
                            <span className={`ml-1 px-2 py-1 rounded text-xs ${
                              district.riskLevel === 'critical' ? 'bg-error-100 text-error-800' :
                              district.riskLevel === 'high' ? 'bg-warning-100 text-warning-800' :
                              district.riskLevel === 'moderate' ? 'bg-accent-100 text-accent-800' :
                              'bg-success-100 text-success-800'
                            }`}>
                              {district.riskLevel.toUpperCase()}
                            </span>
                          </p>
                          <p><strong>Healthcare Facilities:</strong></p>
                          <ul className="text-xs ml-2">
                            {district.facilities.slice(0, 3).map((facility, i) => (
                              <li key={i}>• {facility}</li>
                            ))}
                            {district.facilities.length > 3 && (
                              <li>• +{district.facilities.length - 3} more</li>
                            )}
                          </ul>
                        </div>
                      </div>
                    </Popup>
                  </Circle>
                ))}
                
                {/* Individual detection markers for recent cases */}
                {detectionData.slice(-20).map((detection, index) => (
                  detection.location?.coordinates && (
                    <Marker
                      key={`detection-${index}`}
                      position={[detection.location.coordinates.lat, detection.location.coordinates.lng]}
                    >
                      <Popup>
                        <div className="p-2">
                          <h4 className="font-medium text-medical-800">Recent Diagnosis</h4>
                          <div className="text-sm mt-1 space-y-1">
                            <p><strong>Result:</strong> 
                              <span className={`ml-1 px-2 py-1 rounded text-xs ${
                                detection.result === 'detected' 
                                  ? 'bg-error-100 text-error-800' 
                                  : 'bg-success-100 text-success-800'
                              }`}>
                                {detection.result === 'detected' ? 'POSITIVE' : 'NEGATIVE'}
                              </span>
                            </p>
                            <p><strong>District:</strong> {detection.location.district?.name}</p>
                            <p><strong>Facility:</strong> {detection.location.facility?.name}</p>
                            {detection.location.patientName && (
                              <p><strong>Patient:</strong> {detection.location.patientName}</p>
                            )}
                            <p><strong>Date:</strong> {new Date(detection.timestamp).toLocaleDateString()}</p>
                            <p><strong>ID:</strong> {detection.id}</p>
                          </div>
                        </div>
                      </Popup>
                    </Marker>
                  )
                ))}
              </MapContainer>
            )}
          </div>
        </Card>
      </div>

      {/* Enhanced District Details Table */}
      {districtData.length > 0 && (
        <Card className="overflow-hidden border-l-4 border-l-medical-500">
          <div className="p-4 border-b dark:border-gray-700 bg-medical-50 dark:bg-medical-900/20">
            <h2 className="text-xl font-semibold text-medical-800 dark:text-medical-200 flex items-center">
              <FiBarChart2 className="mr-2" />
              District Health Surveillance Summary
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    District
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Total Cases
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Positive Cases
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Positive Rate
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Risk Level
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Healthcare Facilities
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Last Updated
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {districtData
                  .sort((a, b) => b.totalCases - a.totalCases)
                  .map((district, index) => (
                    <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                      <td className="px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">
                        <div>
                          <div className="font-medium">{district.name}</div>
                          <div className="text-xs text-gray-500">{district.province}</div>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {district.totalCases}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        <span className="text-error-600 font-medium">{district.positiveCases}</span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        {district.positiveRate}%
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <div className="flex items-center">
                          <div 
                            className="w-3 h-3 rounded-full mr-2"
                            style={{ backgroundColor: getCircleColor(district.intensity) }}
                          />
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            district.riskLevel === 'critical' ? 'bg-error-100 text-error-800 dark:bg-error-900/50 dark:text-error-200' :
                            district.riskLevel === 'high' ? 'bg-warning-100 text-warning-800 dark:bg-warning-900/50 dark:text-warning-200' :
                            district.riskLevel === 'moderate' ? 'bg-accent-100 text-accent-800 dark:bg-accent-900/50 dark:text-accent-200' :
                            'bg-success-100 text-success-800 dark:bg-success-900/50 dark:text-success-200'
                          }`}>
                            {district.riskLevel.toUpperCase()}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100">
                        <div className="flex items-center">
                          <FiHome className="mr-1 text-gray-400" size={12} />
                          {district.facilities.length}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                        {new Date(district.lastUpdated).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </motion.div>
  )
}