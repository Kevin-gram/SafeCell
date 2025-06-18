/**
 * Rwanda Administrative Divisions Service
 * Provides access to provinces, districts, and sectors data with healthcare facilities
 */

// Rwanda administrative data with healthcare facilities
const RWANDA_LOCATIONS = {
  provinces: [
    {
      id: 'kigali',
      name: 'Kigali City',
      districts: [
        {
          id: 'gasabo',
          name: 'Gasabo',
          coordinates: { lat: -1.9403, lng: 30.0615 },
          healthFacilities: [
            { id: 'kigali_hospital', name: 'Kigali University Teaching Hospital (CHUK)', type: 'hospital' },
            { id: 'king_faisal', name: 'King Faisal Hospital', type: 'hospital' },
            { id: 'gasabo_hc', name: 'Gasabo Health Center', type: 'health_center' },
            { id: 'kimisagara_hc', name: 'Kimisagara Health Center', type: 'health_center' },
            { id: 'remera_clinic', name: 'Remera Medical Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'bumbogo', name: 'Bumbogo' },
            { id: 'gatsata', name: 'Gatsata' },
            { id: 'gikomero', name: 'Gikomero' },
            { id: 'gisozi', name: 'Gisozi' },
            { id: 'jabana', name: 'Jabana' },
            { id: 'jali', name: 'Jali' },
            { id: 'kacyiru', name: 'Kacyiru' },
            { id: 'kimihurura', name: 'Kimihurura' },
            { id: 'kimisagara', name: 'Kimisagara' },
            { id: 'kinyinya', name: 'Kinyinya' },
            { id: 'ndera', name: 'Ndera' },
            { id: 'nduba', name: 'Nduba' },
            { id: 'remera', name: 'Remera' },
            { id: 'rusororo', name: 'Rusororo' },
            { id: 'rutunga', name: 'Rutunga' }
          ]
        },
        {
          id: 'kicukiro',
          name: 'Kicukiro',
          coordinates: { lat: -1.9659, lng: 30.1032 },
          healthFacilities: [
            { id: 'kanombe_hospital', name: 'Kanombe Military Hospital', type: 'hospital' },
            { id: 'kicukiro_hc', name: 'Kicukiro Health Center', type: 'health_center' },
            { id: 'gatenga_clinic', name: 'Gatenga Medical Clinic', type: 'clinic' },
            { id: 'nyarugunga_hc', name: 'Nyarugunga Health Center', type: 'health_center' }
          ],
          sectors: [
            { id: 'gahanga', name: 'Gahanga' },
            { id: 'gatenga', name: 'Gatenga' },
            { id: 'gikondo', name: 'Gikondo' },
            { id: 'kagarama', name: 'Kagarama' },
            { id: 'kanombe', name: 'Kanombe' },
            { id: 'kicukiro', name: 'Kicukiro' },
            { id: 'masaka', name: 'Masaka' },
            { id: 'niboye', name: 'Niboye' },
            { id: 'nyarugunga', name: 'Nyarugunga' },
            { id: 'ruhango', name: 'Ruhango' }
          ]
        },
        {
          id: 'nyarugenge',
          name: 'Nyarugenge',
          coordinates: { lat: -1.9441, lng: 30.0619 },
          healthFacilities: [
            { id: 'central_hospital', name: 'Central Hospital of Kigali (CHK)', type: 'hospital' },
            { id: 'muhima_hospital', name: 'Muhima District Hospital', type: 'hospital' },
            { id: 'nyamirambo_hc', name: 'Nyamirambo Health Center', type: 'health_center' },
            { id: 'kigali_clinic', name: 'Kigali Medical Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'gitega', name: 'Gitega' },
            { id: 'kanyinya', name: 'Kanyinya' },
            { id: 'kigali', name: 'Kigali' },
            { id: 'kimisagara', name: 'Kimisagara' },
            { id: 'mageragere', name: 'Mageragere' },
            { id: 'muhima', name: 'Muhima' },
            { id: 'nyakabanda', name: 'Nyakabanda' },
            { id: 'nyamirambo', name: 'Nyamirambo' },
            { id: 'nyarugenge', name: 'Nyarugenge' },
            { id: 'rwezamenyo', name: 'Rwezamenyo' }
          ]
        }
      ]
    },
    {
      id: 'eastern',
      name: 'Eastern Province',
      districts: [
        {
          id: 'bugesera',
          name: 'Bugesera',
          coordinates: { lat: -2.2167, lng: 30.2833 },
          healthFacilities: [
            { id: 'nyamata_hospital', name: 'Nyamata District Hospital', type: 'hospital' },
            { id: 'bugesera_hc', name: 'Bugesera Health Center', type: 'health_center' },
            { id: 'rilima_clinic', name: 'Rilima Medical Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'gashora', name: 'Gashora' },
            { id: 'juru', name: 'Juru' },
            { id: 'kamabuye', name: 'Kamabuye' },
            { id: 'ntarama', name: 'Ntarama' },
            { id: 'nyamata', name: 'Nyamata' },
            { id: 'nyarugenge', name: 'Nyarugenge' },
            { id: 'rilima', name: 'Rilima' },
            { id: 'ruhuha', name: 'Ruhuha' },
            { id: 'rweru', name: 'Rweru' }
          ]
        },
        {
          id: 'nyagatare',
          name: 'Nyagatare',
          coordinates: { lat: -1.2938, lng: 30.3275 },
          healthFacilities: [
            { id: 'nyagatare_hospital', name: 'Nyagatare District Hospital', type: 'hospital' },
            { id: 'matimba_hc', name: 'Matimba Health Center', type: 'health_center' },
            { id: 'gatunda_clinic', name: 'Gatunda Border Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'gatunda', name: 'Gatunda' },
            { id: 'karangazi', name: 'Karangazi' },
            { id: 'katabagemu', name: 'Katabagemu' },
            { id: 'kiyombe', name: 'Kiyombe' },
            { id: 'matimba', name: 'Matimba' },
            { id: 'mimuli', name: 'Mimuli' },
            { id: 'mukama', name: 'Mukama' },
            { id: 'musheri', name: 'Musheri' },
            { id: 'nyagatare', name: 'Nyagatare' },
            { id: 'rukomo', name: 'Rukomo' },
            { id: 'rwimiyaga', name: 'Rwimiyaga' },
            { id: 'tabagwe', name: 'Tabagwe' }
          ]
        }
      ]
    },
    {
      id: 'northern',
      name: 'Northern Province',
      districts: [
        {
          id: 'musanze',
          name: 'Musanze',
          coordinates: { lat: -1.4995, lng: 29.6335 },
          healthFacilities: [
            { id: 'ruhengeri_hospital', name: 'Ruhengeri Referral Hospital', type: 'hospital' },
            { id: 'musanze_hc', name: 'Musanze Health Center', type: 'health_center' },
            { id: 'kinigi_clinic', name: 'Kinigi Health Post', type: 'clinic' }
          ],
          sectors: [
            { id: 'busogo', name: 'Busogo' },
            { id: 'cyuve', name: 'Cyuve' },
            { id: 'gacaca', name: 'Gacaca' },
            { id: 'gashaki', name: 'Gashaki' },
            { id: 'gataraga', name: 'Gataraga' },
            { id: 'kimonyi', name: 'Kimonyi' },
            { id: 'kinigi', name: 'Kinigi' },
            { id: 'muhoza', name: 'Muhoza' },
            { id: 'muko', name: 'Muko' },
            { id: 'musanze', name: 'Musanze' },
            { id: 'nkotsi', name: 'Nkotsi' },
            { id: 'nyange', name: 'Nyange' },
            { id: 'remera', name: 'Remera' },
            { id: 'rwaza', name: 'Rwaza' },
            { id: 'shingiro', name: 'Shingiro' }
          ]
        },
        {
          id: 'burera',
          name: 'Burera',
          coordinates: { lat: -1.4667, lng: 29.8833 },
          healthFacilities: [
            { id: 'butaro_hospital', name: 'Butaro Cancer Center of Excellence', type: 'hospital' },
            { id: 'burera_hc', name: 'Burera Health Center', type: 'health_center' },
            { id: 'cyanika_clinic', name: 'Cyanika Border Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'bungwe', name: 'Bungwe' },
            { id: 'butaro', name: 'Butaro' },
            { id: 'cyanika', name: 'Cyanika' },
            { id: 'cyeru', name: 'Cyeru' },
            { id: 'gahunga', name: 'Gahunga' },
            { id: 'gatebe', name: 'Gatebe' },
            { id: 'gitovu', name: 'Gitovu' },
            { id: 'kagogo', name: 'Kagogo' },
            { id: 'kinoni', name: 'Kinoni' },
            { id: 'kinyababa', name: 'Kinyababa' },
            { id: 'kivuye', name: 'Kivuye' },
            { id: 'nemba', name: 'Nemba' },
            { id: 'rugarama', name: 'Rugarama' },
            { id: 'rugendabari', name: 'Rugendabari' },
            { id: 'ruhunde', name: 'Ruhunde' },
            { id: 'rusarabuye', name: 'Rusarabuye' },
            { id: 'rwerere', name: 'Rwerere' }
          ]
        }
      ]
    },
    {
      id: 'southern',
      name: 'Southern Province',
      districts: [
        {
          id: 'huye',
          name: 'Huye',
          coordinates: { lat: -2.6399, lng: 29.7406 },
          healthFacilities: [
            { id: 'butare_hospital', name: 'Butare University Teaching Hospital (CHUB)', type: 'hospital' },
            { id: 'huye_hc', name: 'Huye Health Center', type: 'health_center' },
            { id: 'tumba_clinic', name: 'Tumba College Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'gishamvu', name: 'Gishamvu' },
            { id: 'karama', name: 'Karama' },
            { id: 'kigoma', name: 'Kigoma' },
            { id: 'kinazi', name: 'Kinazi' },
            { id: 'maraba', name: 'Maraba' },
            { id: 'mbazi', name: 'Mbazi' },
            { id: 'mukura', name: 'Mukura' },
            { id: 'ngoma', name: 'Ngoma' },
            { id: 'ruhashya', name: 'Ruhashya' },
            { id: 'rusatira', name: 'Rusatira' },
            { id: 'rwaniro', name: 'Rwaniro' },
            { id: 'simbi', name: 'Simbi' },
            { id: 'tumba', name: 'Tumba' }
          ]
        },
        {
          id: 'nyanza',
          name: 'Nyanza',
          coordinates: { lat: -2.3500, lng: 29.7500 },
          healthFacilities: [
            { id: 'nyanza_hospital', name: 'Nyanza District Hospital', type: 'hospital' },
            { id: 'busoro_hc', name: 'Busoro Health Center', type: 'health_center' },
            { id: 'mukingo_clinic', name: 'Mukingo Health Post', type: 'clinic' }
          ],
          sectors: [
            { id: 'busasamana', name: 'Busasamana' },
            { id: 'busoro', name: 'Busoro' },
            { id: 'cyabakamyi', name: 'Cyabakamyi' },
            { id: 'kibirizi', name: 'Kibirizi' },
            { id: 'kigoma', name: 'Kigoma' },
            { id: 'mukingo', name: 'Mukingo' },
            { id: 'muyira', name: 'Muyira' },
            { id: 'ntyazo', name: 'Ntyazo' },
            { id: 'nyagisozi', name: 'Nyagisozi' },
            { id: 'rwabicuma', name: 'Rwabicuma' }
          ]
        }
      ]
    },
    {
      id: 'western',
      name: 'Western Province',
      districts: [
        {
          id: 'rubavu',
          name: 'Rubavu',
          coordinates: { lat: -1.6777, lng: 29.2505 },
          healthFacilities: [
            { id: 'gisenyi_hospital', name: 'Gisenyi District Hospital', type: 'hospital' },
            { id: 'rubavu_hc', name: 'Rubavu Health Center', type: 'health_center' },
            { id: 'mudende_clinic', name: 'Mudende Refugee Camp Clinic', type: 'clinic' }
          ],
          sectors: [
            { id: 'bugeshi', name: 'Bugeshi' },
            { id: 'busasamana', name: 'Busasamana' },
            { id: 'cyanzarwe', name: 'Cyanzarwe' },
            { id: 'gisenyi', name: 'Gisenyi' },
            { id: 'kanama', name: 'Kanama' },
            { id: 'kanzenze', name: 'Kanzenze' },
            { id: 'mudende', name: 'Mudende' },
            { id: 'nyakiliba', name: 'Nyakiliba' },
            { id: 'nyakiriba', name: 'Nyakiriba' },
            { id: 'nyundo', name: 'Nyundo' },
            { id: 'rubavu', name: 'Rubavu' },
            { id: 'rugerero', name: 'Rugerero' }
          ]
        },
        {
          id: 'karongi',
          name: 'Karongi',
          coordinates: { lat: -1.9500, lng: 29.3833 },
          healthFacilities: [
            { id: 'kibuye_hospital', name: 'Kibuye Hope Hospital', type: 'hospital' },
            { id: 'karongi_hc', name: 'Karongi Health Center', type: 'health_center' },
            { id: 'bwishyura_clinic', name: 'Bwishyura Health Post', type: 'clinic' }
          ],
          sectors: [
            { id: 'bwishyura', name: 'Bwishyura' },
            { id: 'gashari', name: 'Gashari' },
            { id: 'gitesi', name: 'Gitesi' },
            { id: 'gishyita', name: 'Gishyita' },
            { id: 'gisovu', name: 'Gisovu' },
            { id: 'kivumu', name: 'Kivumu' },
            { id: 'mahembe', name: 'Mahembe' },
            { id: 'murambi', name: 'Murambi' },
            { id: 'murundi', name: 'Murundi' },
            { id: 'mutuntu', name: 'Mutuntu' },
            { id: 'rugabano', name: 'Rugabano' },
            { id: 'ruganda', name: 'Ruganda' },
            { id: 'rwankuba', name: 'Rwankuba' }
          ]
        }
      ]
    }
  ]
}

/**
 * Get all provinces
 */
export const getProvinces = () => {
  return RWANDA_LOCATIONS.provinces.map(province => ({
    value: province.id,
    label: province.name
  }))
}

/**
 * Get districts for a specific province
 */
export const getDistricts = (provinceId) => {
  const province = RWANDA_LOCATIONS.provinces.find(p => p.id === provinceId)
  if (!province) return []
  
  return province.districts.map(district => ({
    value: district.id,
    label: district.name
  }))
}

/**
 * Get sectors for a specific district
 */
export const getSectors = (provinceId, districtId) => {
  const province = RWANDA_LOCATIONS.provinces.find(p => p.id === provinceId)
  if (!province) return []
  
  const district = province.districts.find(d => d.id === districtId)
  if (!district) return []
  
  return district.sectors.map(sector => ({
    value: sector.id,
    label: sector.name
  }))
}

/**
 * Get health facilities for a specific district
 */
export const getHealthFacilities = (provinceId, districtId) => {
  const province = RWANDA_LOCATIONS.provinces.find(p => p.id === provinceId)
  if (!province) return []
  
  const district = province.districts.find(d => d.id === districtId)
  if (!district) return []
  
  return district.healthFacilities.map(facility => ({
    value: facility.id,
    label: `${facility.name} (${facility.type.replace('_', ' ').toUpperCase()})`
  }))
}

/**
 * Get location details by IDs
 */
export const getLocationDetails = (provinceId, districtId, sectorId, facilityId = null) => {
  const province = RWANDA_LOCATIONS.provinces.find(p => p.id === provinceId)
  if (!province) return null
  
  const district = province.districts.find(d => d.id === districtId)
  if (!district) return null
  
  const sector = district.sectors.find(s => s.id === sectorId)
  if (!sector) return null
  
  let facility = null
  if (facilityId) {
    facility = district.healthFacilities.find(f => f.id === facilityId)
  }
  
  return {
    province: { id: province.id, name: province.name },
    district: { id: district.id, name: district.name, coordinates: district.coordinates },
    sector: { id: sector.id, name: sector.name },
    facility: facility ? { id: facility.id, name: facility.name, type: facility.type } : null
  }
}

/**
 * Get coordinates for a district (more accurate than province-level)
 */
export const getDistrictCoordinates = (provinceId, districtId) => {
  const province = RWANDA_LOCATIONS.provinces.find(p => p.id === provinceId)
  if (!province) return { lat: -1.9403, lng: 29.8739 }
  
  const district = province.districts.find(d => d.id === districtId)
  return district?.coordinates || { lat: -1.9403, lng: 29.8739 }
}

/**
 * Get all districts with their coordinates for mapping
 */
export const getAllDistrictsWithCoordinates = () => {
  const districts = []
  
  RWANDA_LOCATIONS.provinces.forEach(province => {
    province.districts.forEach(district => {
      districts.push({
        id: district.id,
        name: district.name,
        province: province.name,
        coordinates: district.coordinates
      })
    })
  })
  
  return districts
}