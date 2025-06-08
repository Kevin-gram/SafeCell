import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../contexts/I18nContext'
import { MapContainer, TileLayer, Circle, Popup } from 'react-leaflet'
import Card from '../components/ui/Card'
import Select from '../components/ui/Select'
import { FiMapPin, FiAlertTriangle } from 'react-icons/fi'
import 'leaflet/dist/leaflet.css'

// Rwanda's center coordinates and bounds
const RWANDA_CENTER = [-1.9403, 29.8739]
const RWANDA_BOUNDS = [
  [-2.8389, 28.8617], // Southwest
  [-1.0474, 30.8862]  // Northeast
]

// Intensity levels with descriptions
const INTENSITY_LEVELS = [
  { level: 1, name: 'Low', color: '#10B981', description: 'Few parasites detected (1-10 per field)' },
  { level: 2, name: 'Mild', color: '#FBBF24', description: 'Moderate parasite count (11-100 per field)' },
  { level: 3, name: 'Moderate', color: '#F59E0B', description: 'Significant parasite presence (101-500 per field)' },
  { level: 4, name: 'High', color: '#DC2626', description: 'Heavy parasite load (501-1000 per field)' },
  { level: 5, name: 'Severe', color: '#991B1B', description: 'Extreme parasite density (>1000 per field)' }
]

export default function LocationStats() {
  const { t } = useI18n()
  const [timeRange, setTimeRange] = useState('month')
  const [loading, setLoading] = useState(true)
  const [locationData, setLocationData] = useState([])
  
  // Mock data for demonstration
  useEffect(() => {
    const fetchLocationData = async () => {
      setLoading(true)
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        // Mock data for different locations in Rwanda
        const mockData = [
          {
            name: 'Kigali',
            coordinates: [-1.9474, 30.0615],
            cases: 245,
            intensity: 4
          },
          {
            name: 'Musanze',
            coordinates: [-1.4995, 29.6335],
            cases: 156,
            intensity: 3
          },
          {
            name: 'Rubavu',
            coordinates: [-1.6777, 29.2505],
            cases: 89,
            intensity: 2
          },
          {
            name: 'Nyagatare',
            coordinates: [-1.2938, 30.3275],
            cases: 178,
            intensity: 3
          },
          {
            name: 'Huye',
            coordinates: [-2.6399, 29.7406],
            cases: 267,
            intensity: 4
          }
        ]
        
        setLocationData(mockData)
      } catch (error) {
        console.error('Failed to fetch location data:', error)
      } finally {
        setLoading(false)
      }
    }
    
    fetchLocationData()
  }, [timeRange])
  
  const getCircleColor = (intensity) => {
    return INTENSITY_LEVELS[intensity - 1]?.color || '#000000'
  }
  
  const getCircleSize = (cases) => {
    return Math.sqrt(cases) * 500 // Scale circle size based on number of cases
  }
  
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-8"
    >
      <div>
        <h1 className="text-2xl md:text-3xl font-bold flex items-center">
          <FiMapPin className="mr-2" />
          {t('statistics.locationStats')}
        </h1>
        <p className="text-gray-600 dark:text-gray-400 mt-2">
          Malaria distribution and intensity levels across Rwanda
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Intensity Levels Card */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <FiAlertTriangle className="mr-2" />
            Intensity Levels
          </h2>
          <div className="space-y-4">
            {INTENSITY_LEVELS.map(({ level, name, color, description }) => (
              <div key={level} className="flex items-start space-x-3">
                <div 
                  className="w-4 h-4 rounded-full mt-1 flex-shrink-0"
                  style={{ backgroundColor: color }}
                />
                <div>
                  <h3 className="font-medium">Level {level} - {name}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
        
        {/* Map Card */}
        <Card className="lg:col-span-2 overflow-hidden">
          <div className="h-[600px]">
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
              
              {locationData.map((location) => (
                <Circle
                  key={location.name}
                  center={location.coordinates}
                  radius={getCircleSize(location.cases)}
                  pathOptions={{
                    color: getCircleColor(location.intensity),
                    fillColor: getCircleColor(location.intensity),
                    fillOpacity: 0.6
                  }}
                >
                  <Popup>
                    <div className="p-2">
                      <h3 className="font-bold">{location.name}</h3>
                      <p>Cases: {location.cases}</p>
                      <p>Intensity: Level {location.intensity}</p>
                    </div>
                  </Popup>
                </Circle>
              ))}
            </MapContainer>
          </div>
        </Card>
      </div>
    </motion.div>
  )
}