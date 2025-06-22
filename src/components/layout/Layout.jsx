import { useState, useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from './Navbar'
import Sidebar from './Sidebar'

export default function Layout() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkSize = () => {
      setIsMobile(window.innerWidth < 1024)
      if (window.innerWidth >= 1024) {
        setIsSidebarOpen(true)
      } else {
        setIsSidebarOpen(false)
      }
    }

    // Set initial size
    checkSize()

    // Add event listener
    window.addEventListener('resize', checkSize)

    // Clean up
    return () => window.removeEventListener('resize', checkSize)
  }, [])

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen)
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar toggleSidebar={toggleSidebar} isSidebarOpen={isSidebarOpen} />
      
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar for larger screens */}
        {!isMobile && (
          <AnimatePresence>
            {isSidebarOpen && (
              <motion.div 
                initial={{ width: 0, opacity: 0 }}
                animate={{ width: 256, opacity: 1 }}
                exit={{ width: 0, opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="hidden lg:block"
              >
                <Sidebar closeSidebar={() => isMobile && setIsSidebarOpen(false)} />
              </motion.div>
            )}
          </AnimatePresence>
        )}
        
        {/* Mobile sidebar overlay */}
        {isMobile && (
          <AnimatePresence>
            {isSidebarOpen && (
              <>
                <motion.div
                  className="fixed inset-0 bg-black bg-opacity-50 z-20"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={() => setIsSidebarOpen(false)}
                />
                <motion.div
                  className="fixed left-0 top-0 bottom-0 w-64 z-30"
                  initial={{ x: '-100%' }}
                  animate={{ x: 0 }}
                  exit={{ x: '-100%' }}
                  transition={{ type: 'spring', damping: 25 }}
                >
                  <Sidebar closeSidebar={() => setIsSidebarOpen(false)} />
                </motion.div>
              </>
            )}
          </AnimatePresence>
        )}
        
        {/* Main content */}
        <main className={`flex-1 overflow-auto transition-all duration-300 ${isSidebarOpen && !isMobile ? 'ml-16' : 'ml-0'}`}>
          <div className="container-custom py-6 px-4 sm:px-6 lg:px-8 min-h-full">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}