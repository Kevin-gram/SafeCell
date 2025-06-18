import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useI18n } from '../../contexts/I18nContext'
import { useLogger } from '../../hooks/useLogger'
import Card from '../ui/Card'
import Select from '../ui/Select'
import Input from '../ui/Input'
import { FiMapPin, FiAlertCircle, FiUser, FiHome } from 'react-icons/fi'
import { 
  getProvinces, 
  getDistricts, 
  getSectors, 
  getHealthFacilities,
  getLocationDetails 
} from '../../services/rwandaLocations'

export default function LocationSelector({ 
  selectedLocation, 
  onLocationChange, 
  error,
  required = true 
}) {
  const { t } = useI18n()
  const { logInteraction } = useLogger()
  
  const [provinces] = useState(getProvinces())
  const [districts, setDistricts] = useState([])
  const [sectors, setSectors] = useState([])
  const [healthFacilities, setHealthFacilities] = useState([])
  const [locationData, setLocationData] = useState({
    province: '',
    district: '',
    sector: '',
    facility: '',
    patientName: ''
  })

  // Update districts when province changes
  useEffect(() => {
    if (locationData.province) {
      const districtOptions = getDistricts(locationData.province)
      setDistricts(districtOptions)
      
      // Reset dependent fields if province changed
      if (locationData.district && !districtOptions.find(d => d.value === locationData.district)) {
        setLocationData(prev => ({ 
          ...prev, 
          district: '', 
          sector: '', 
          facility: '' 
        }))
        setSectors([])
        setHealthFacilities([])
      }
    } else {
      setDistricts([])
      setSectors([])
      setHealthFacilities([])
    }
  }, [locationData.province])

  // Update sectors and health facilities when district changes
  useEffect(() => {
    if (locationData.province && locationData.district) {
      const sectorOptions = getSectors(locationData.province, locationData.district)
      setSectors(sectorOptions)
      
      const facilityOptions = getHealthFacilities(locationData.province, locationData.district)
      setHealthFacilities(facilityOptions)
      
      // Reset dependent fields if district changed
      if (locationData.sector && !sectorOptions.find(s => s.value === locationData.sector)) {
        setLocationData(prev => ({ ...prev, sector: '' }))
      }
      if (locationData.facility && !facilityOptions.find(f => f.value === locationData.facility)) {
        setLocationData(prev => ({ ...prev, facility: '' }))
      }
    } else {
      setSectors([])
      setHealthFacilities([])
    }
  }, [locationData.province, locationData.district])

  // Notify parent component when location is complete
  useEffect(() => {
    if (locationData.province && locationData.district && locationData.sector && locationData.facility) {
      const details = getLocationDetails(
        locationData.province, 
        locationData.district, 
        locationData.sector,
        locationData.facility
      )
      
      if (details) {
        const completeLocation = {
          ...details,
          patientName: locationData.patientName.trim() || null
        }
        
        onLocationChange(completeLocation)
        
        // Log location selection
        logInteraction('select', 'complete-location', {
          province: details.province.name,
          district: details.district.name,
          sector: details.sector.name,
          facility: details.facility.name,
          hasPatientName: !!locationData.patientName.trim()
        })
      }
    } else {
      onLocationChange(null)
    }
  }, [locationData, onLocationChange, logInteraction])

  // Initialize with selected location if provided
  useEffect(() => {
    if (selectedLocation) {
      setLocationData({
        province: selectedLocation.province.id,
        district: selectedLocation.district.id,
        sector: selectedLocation.sector.id,
        facility: selectedLocation.facility?.id || '',
        patientName: selectedLocation.patientName || ''
      })
    }
  }, [selectedLocation])

  const handleLocationChange = (field, value) => {
    setLocationData(prev => {
      const newData = { ...prev, [field]: value }
      
      // Reset dependent fields when parent field changes
      if (field === 'province') {
        newData.district = ''
        newData.sector = ''
        newData.facility = ''
      } else if (field === 'district') {
        newData.sector = ''
        newData.facility = ''
      }
      
      return newData
    })
  }

  const isComplete = locationData.province && locationData.district && 
                   locationData.sector && locationData.facility

  return (
    <Card className="p-6 border-l-4 border-l-medical-500">
      <div className="flex items-center mb-4">
        <FiMapPin className="text-medical-600 dark:text-medical-400 mr-2" size={20} />
        <h3 className="text-lg font-medium text-medical-800 dark:text-medical-200">
          Diagnosis Location {required && <span className="text-error-500">*</span>}
        </h3>
      </div>
      
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-6">
        Please provide complete location details and healthcare facility information for accurate regional data collection and mapping.
      </p>

      <div className="space-y-4">
        {/* Patient Name - Optional */}
        <div className="bg-medical-50 dark:bg-medical-900/20 p-4 rounded-lg border border-medical-200 dark:border-medical-800">
          <div className="flex items-center mb-2">
            <FiUser className="text-medical-600 dark:text-medical-400 mr-2" size={16} />
            <h4 className="text-sm font-medium text-medical-700 dark:text-medical-300">
              Patient Information
            </h4>
          </div>
          <Input
            label="Patient Name (Optional)"
            value={locationData.patientName}
            onChange={(e) => handleLocationChange('patientName', e.target.value)}
            placeholder="Enter patient's name (optional)"
            className="bg-white dark:bg-gray-700"
          />
          <p className="text-xs text-medical-600 dark:text-medical-400 mt-1">
            Patient name is optional and will be kept confidential for medical records.
          </p>
        </div>

        {/* Administrative Location */}
        <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
          <div className="flex items-center mb-3">
            <FiMapPin className="text-blue-600 dark:text-blue-400 mr-2" size={16} />
            <h4 className="text-sm font-medium text-blue-700 dark:text-blue-300">
              Administrative Location
            </h4>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Select
              label={t('statistics.province')}
              value={locationData.province}
              onChange={(e) => handleLocationChange('province', e.target.value)}
              options={[
                { value: '', label: 'Select Province' },
                ...provinces
              ]}
              required={required}
              error={error && !locationData.province ? 'Province is required' : ''}
            />

            <Select
              label={t('statistics.district')}
              value={locationData.district}
              onChange={(e) => handleLocationChange('district', e.target.value)}
              options={[
                { value: '', label: locationData.province ? 'Select District' : 'Select Province First' },
                ...districts
              ]}
              disabled={!locationData.province}
              required={required}
              error={error && locationData.province && !locationData.district ? 'District is required' : ''}
            />

            <Select
              label={t('statistics.sector')}
              value={locationData.sector}
              onChange={(e) => handleLocationChange('sector', e.target.value)}
              options={[
                { value: '', label: locationData.district ? 'Select Sector' : 'Select District First' },
                ...sectors
              ]}
              disabled={!locationData.district}
              required={required}
              error={error && locationData.district && !locationData.sector ? 'Sector is required' : ''}
            />
          </div>
        </div>

        {/* Healthcare Facility */}
        <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
          <div className="flex items-center mb-3">
            <FiHome className="text-green-600 dark:text-green-400 mr-2" size={16} />
            <h4 className="text-sm font-medium text-green-700 dark:text-green-300">
              Healthcare Facility
            </h4>
          </div>
          
          <Select
            label="Hospital/Clinic/Health Center"
            value={locationData.facility}
            onChange={(e) => handleLocationChange('facility', e.target.value)}
            options={[
              { value: '', label: locationData.district ? 'Select Healthcare Facility' : 'Select District First' },
              ...healthFacilities
            ]}
            disabled={!locationData.district}
            required={required}
            error={error && locationData.district && !locationData.facility ? 'Healthcare facility is required' : ''}
          />
          <p className="text-xs text-green-600 dark:text-green-400 mt-1">
            Select the hospital, clinic, or health center where the diagnosis is being performed.
          </p>
        </div>
      </div>

      {error && !isComplete && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-md"
        >
          <div className="flex items-start">
            <FiAlertCircle className="text-error-500 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm text-error-700 dark:text-error-300 font-medium">
                Complete Location Required
              </p>
              <p className="text-sm text-error-600 dark:text-error-400 mt-1">
                Please select your complete location (Province, District, Sector, and Healthcare Facility) before proceeding with the diagnosis.
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {isComplete && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-3 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-md"
        >
          <div className="flex items-start">
            <FiMapPin className="text-success-500 mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="text-sm text-success-700 dark:text-success-300 font-medium">
                Location Complete
              </p>
              <div className="text-sm text-success-600 dark:text-success-400 mt-1">
                <p><strong>Location:</strong> {selectedLocation?.sector.name}, {selectedLocation?.district.name}, {selectedLocation?.province.name}</p>
                <p><strong>Facility:</strong> {selectedLocation?.facility?.name}</p>
                {selectedLocation?.patientName && (
                  <p><strong>Patient:</strong> {selectedLocation.patientName}</p>
                )}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </Card>
  )
}