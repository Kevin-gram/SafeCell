import { motion } from 'framer-motion'

export default function Card({ 
  children, 
  className = '', 
  hoverable = false,
  onClick = null,
  animate = true
}) {
  const variants = {
    initial: { opacity: 0, y: 10 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.4 } },
    hover: { 
      y: -5, 
      boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
      transition: { duration: 0.2 }
    }
  }

  return (
    <motion.div
      className={`bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden ${hoverable ? 'cursor-pointer' : ''} ${className}`}
      initial={animate ? "initial" : false}
      animate={animate ? "animate" : false}
      whileHover={hoverable ? "hover" : false}
      onClick={onClick}
      variants={variants}
    >
      {children}
    </motion.div>
  )
}