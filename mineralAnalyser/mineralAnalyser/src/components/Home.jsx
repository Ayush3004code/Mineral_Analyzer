// Home.jsx
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative bg-[url('./src/assets/img12.jpg')] bg-cover bg-center text-white py-20 mt-16">
      {/* <section className="bg-gradient-to-r from-blue-900 to-gray-900 text-white py-20 mt-16"> */}
      <div className="absolute inset-0 bg-black opacity-50"></div>
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Mineral Analysis Made Simple
            </h1>
            <p className="text-xl mb-8">
              Identify rocks and minerals instantly using our advanced AI technology
            </p>
            <Link
              to="/analyzer"
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-full transition duration-300"
            >
              Start Analyzing
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Our Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-blue-500 mb-4">
                <svg
                  className="w-12 h-12"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-bold mb-2">Instant Analysis</h3>
              <p className="text-gray-600">
                Get quick and accurate mineral identification within seconds using our
                advanced AI algorithms.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-blue-500 mb-4">
                <svg
                  className="w-12 h-12"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                  />
                </svg>
              </div >
              <h3 className="text-xl font-bold mb-2">Comprehensive Database</h3>
              <p className="text-gray-600">
                Our database contains a vast collection of minerals and rocks, ensuring
                accurate identification and analysis.
              </p>
            </div>

            <div className="bg-white p-6 rounded-lg shadow-md">
              <div className="text-blue-500 mb-4">
                <svg
                  className="w-12 h-12"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v-6a2 2 0 012-2m-1-4H9a2 2 0 00-2 2v6a2 2 0 002 2h6a2 2 0 002-2V10a2 2 0 00-2-2m-1-4H9a2 2 0 00-2 2v6a2 2 0 002 2h6a2 2 0 002-2V10a2 2 0 00-2-2"
                  />
                </svg>
              </div>
              <h3 className="text-xl font-bold mb-2">User-Friendly Interface</h3>
              <p className="text-gray-600">
                Our intuitive interface makes it easy for users of all levels to navigate
                and analyzer rocks and minerals.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Call to Action Section */}
      <section className="bg-gray-900 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">
            Get Started with Mineral Analysis Today!
          </h2>
          <p className="text-xl text-center mb-8">
            Join the thousands of geologists, researchers, and enthusiasts who trust
            Mineral Analyser for their rock and mineral analysis needs.
          </p>
          <div className="text-center">
            <Link
              to="/analyzer"
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-full transition duration-300"
            >
              Start Analyzing
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;