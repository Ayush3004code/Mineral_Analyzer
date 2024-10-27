// Header.jsx
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const location = useLocation();

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const isActiveLink = (path) => location.pathname === path;

  return (
    <header className="fixed top-0 left-0 right-0 bg-gray-900/95 backdrop-blur-sm text-white shadow-lg z-50 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex-shrink-0">
            <Link 
              to="/" 
              className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent hover:from-blue-500 hover:to-purple-600 transition-all duration-300 transform hover:scale-105"
            >
              Mineral Analyser
            </Link>
          </div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex space-x-6">
            {[
              { to: '/analyzer', text: 'Analyze Rock' },
              { to: '/login', text: 'Login' },
              { to: '/signup', text: 'Sign Up' },
              { to: '/about', text: 'About Us' }
            ].map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ease-in-out
                  ${isActiveLink(link.to)
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg shadow-blue-500/25'
                    : 'hover:bg-gray-800 hover:shadow-md hover:shadow-purple-500/10 hover:text-blue-400'
                  } relative overflow-hidden group`}
              >
                <span className="relative z-10">{link.text}</span>
                <div className="absolute inset-0 h-full w-full bg-gradient-to-r from-blue-600 to-purple-600 opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              </Link>
            ))}
          </nav>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              onClick={toggleMenu}
              className="inline-flex items-center justify-center p-2 rounded-lg hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
            >
              <span className="sr-only">Open main menu</span>
              {/* Hamburger icon */}
              <svg
                className={`${isMenuOpen ? 'hidden' : 'block'} h-6 w-6 text-blue-400`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              {/* Close icon */}
              <svg
                className={`${isMenuOpen ? 'block' : 'hidden'} h-6 w-6 text-blue-400`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`${isMenuOpen ? 'block' : 'hidden'} md:hidden`}>
        <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
          {[
            { to: '/analyzer', text: 'Analyze Rock' },
            { to: '/login', text: 'Login' },
            { to: '/signup', text: 'Sign Up' },
            { to: '/about', text: 'About Us' }
          ].map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className="block px-4 py-2 rounded-lg text-base font-medium transition-all duration-300 ease-in-out hover:bg-gray-800 hover:text-blue-400"
            >
              {link.text}
            </Link>
          ))}
        </div>
      </div>
    </header>
  );
};

export default Header;


// // Header.jsx
// import { useState } from 'react';
// import { Link, useLocation } from 'react-router-dom';

// const Header = () => {
//   const [isMenuOpen, setIsMenuOpen] = useState(false);
//   const location = useLocation();

//   const toggleMenu = () => {
//     setIsMenuOpen(!isMenuOpen);
//   };

//   const isActiveLink = (path) => location.pathname === path;

//   return (
//     <header className="fixed top-0 left-0 right-0 bg-gray-900 text-white shadow-lg z-50">
//       <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
//         <div className="flex justify-between items-center h-16">
//             {/* Logo */}
//             <div className="flex-shrink-0">
//                 <Link to="/" className="text-xl font-bold">
//                 Mineral Analyser
//                 </Link>
//             </div>

//             {/* Desktop Navigation */}
//             <nav className="hidden md:flex space-x-4">
//                 <Link
//                 to="/analyzer"
//                 className={`px-3 py-2 rounded-md text-sm font-medium transition duration-150 ease-in-out
//                     ${isActiveLink('/analyzer') 
//                     ? 'bg-gray-700 text-white' 
//                     : 'hover:bg-gray-700'}`}
//                 >
//                 Analyze Rock
//                 </Link>
//                 <Link
//                 to="/login"
//                 className={`px-3 py-2 rounded-md text-sm font-medium transition duration-150 ease-in-out
//                     ${isActiveLink('/login') 
//                     ? 'bg-gray-700 text-white' 
//                     : 'hover:bg-gray-700'}`}
//                 >
//                 Login
//                 </Link>
//                 <Link
//                 to="/signup"
//                 className={`px-3 py-2 rounded-md text-sm font-medium transition duration-150 ease-in-out
//                     ${isActiveLink('/signup') 
//                     ? 'bg-gray-700 text-white' 
//                     : 'hover:bg-gray-700'}`}
//                 >
//                 Sign Up
//                 </Link>
//                 <Link
//                 to="/about"
//                 className={`px-3 py-2 rounded-md text-sm font-medium transition duration-150 ease-in-out
//                     ${isActiveLink('/about') 
//                     ? 'bg-gray-700 text-white' 
//                     : 'hover:bg-gray-700'}`}
//                 >
//                 About Us
//                 </Link>
//             </nav>

//           {/* Mobile menu button */}
//           <div className="md:hidden">
//             <button
//               onClick={toggleMenu}
//               className="inline-flex items-center justify-center p-2 rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
//             >
//               <span className="sr-only">Open main menu</span>
//               {/* Hamburger icon */}
//               <svg
//                 className={`${isMenuOpen ? 'hidden' : 'block'} h-6 w-6`}
//                 xmlns="http://www.w3.org/2000/svg"
//                 fill="none"
//                 viewBox="0 0 24 24"
//                 stroke="currentColor"
//                 aria-hidden="true"
//               >
//                 <path
//                   strokeLinecap="round"
//                   strokeLinejoin="round"
//                   strokeWidth={2}
//                   d="M4 6h16M4 12h16M4 18h16"
//                 />
//               </svg>
//               {/* Close icon */}
//               <svg
//                 className={`${isMenuOpen ? 'block' : 'hidden'} h-6 w-6`}
//                 xmlns="http://www.w3.org/2000/svg"
//                 fill="none"
//                 viewBox="0 0 24 24"
//                 stroke="currentColor"
//                 aria-hidden="true"
//               >
//                 <path
//                   strokeLinecap="round"
//                   strokeLinejoin="round"
//                   strokeWidth={2}
//                   d="M6 18L18 6M6 6l12 12"
//                 />
//               </svg>
//             </button>
//           </div>
//         </div>
//       </div>

//       {/* Mobile menu */}
//       <div className={`${isMenuOpen ? 'block' : 'hidden'} md:hidden`}>
//         <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
//           <Link
//             to="/analyzer" className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-700 transition duration-150 ease-in-out"
//           >
//             Analyze Rock
//           </Link>
//           <Link
//             to="/login"
//             className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-700 transition duration-150 ease-in-out"
//           >
//             Login
//           </Link>
//           <Link
//             to="/signup"
//             className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-700 transition duration-150 ease-in-out"
//           >
//             Sign Up
//           </Link>
//           <Link
//             to="/about"
//             className="block px-3 py-2 rounded-md text-base font-medium hover:bg-gray-700 transition duration-150 ease-in-out"
//           >
//             About Us
//           </Link>
//         </div>
//       </div>
//     </header>
//   );
// };

// export default Header;