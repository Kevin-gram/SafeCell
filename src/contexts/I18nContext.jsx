import { createContext, useContext, useState, useEffect } from 'react'
import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'
import { translationEN } from '../i18n/en'
import { translationFR } from '../i18n/fr'
import { translationRW } from '../i18n/rw'

// Initialize i18next
i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: {
        translation: translationEN
      },
      fr: {
        translation: translationFR
      },
      rw: {
        translation: translationRW
      }
    },
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  })

const I18nContext = createContext(null)

export const I18nProvider = ({ children }) => {
  const [language, setLanguage] = useState(i18n.language || 'en')
  
  // Change language function
  const changeLanguage = (lang) => {
    i18n.changeLanguage(lang)
    setLanguage(lang)
    localStorage.setItem('language', lang)
  }
  
  // Check for saved language preference on mount
  useEffect(() => {
    const savedLanguage = localStorage.getItem('language')
    if (savedLanguage) {
      changeLanguage(savedLanguage)
    }
  }, [])
  
  const value = {
    language,
    changeLanguage,
    t: (key, options) => i18n.t(key, options)
  }
  
  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  )
}

export const useI18n = () => {
  const context = useContext(I18nContext)
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider')
  }
  return context
}