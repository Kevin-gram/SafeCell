import { NavLink } from 'react-router-dom'
import { useI18n } from '../../contexts/I18nContext'
import { FiHome, FiSearch, FiBarChart2, FiMessageSquare, FiSettings, FiMapPin } from 'react-icons/fi'

export default function Sidebar({ closeSidebar }) {
  const { t } = useI18n()

  const navItems = [
    { 
      to: '/', 
      icon: <FiHome size={20} />, 
      label: t('nav.home'),
      exact: true
    },
    { 
      to: '/detection', 
      icon: <FiSearch size={20} />, 
      label: t('nav.detection')
    },
    { 
      to: '/statistics', 
      icon: <FiBarChart2 size={20} />, 
      label: t('nav.statistics')
    },
    {
      to: '/location-stats',
      icon: <FiMapPin size={20} />,
      label: t('nav.locationStats')
    },
    {
      to: '/settings',
      icon: <FiSettings size={20} />,
      label: t('nav.settings')
    },
    { 
      to: '/feedback', 
      icon: <FiMessageSquare size={20} />, 
      label: t('nav.feedback')
    }
  ]

  return (
    <div className="w-64 h-full bg-white dark:bg-gray-800 border-r dark:border-gray-700 shadow-sm flex flex-col">
      <div className="p-4 border-b dark:border-gray-700">
        <h1 className="text-xl font-bold text-primary-600 dark:text-primary-400">{t('common.appName')}</h1>
      </div>
      
      <nav className="flex-1 overflow-y-auto py-4">
        <ul className="space-y-1">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.exact}
                className={({ isActive }) => 
                  `flex items-center px-4 py-3 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ${
                    isActive ? 'bg-primary-50 dark:bg-primary-900/40 text-primary-600 dark:text-primary-400 font-medium border-l-4 border-primary-500' : ''
                  }`
                }
                onClick={closeSidebar}
              >
                <span className="mr-3">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      
      <div className="p-4 border-t dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
        <div className="text-xs text-gray-500 dark:text-gray-400">
          <p>SafeCell v0.1</p>
          <p className="mt-1">© 2025 HealthTech AI</p>
        </div>
      </div>
    </div>
  )
}