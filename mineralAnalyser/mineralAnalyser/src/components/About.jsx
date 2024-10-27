// About.jsx
import { Link } from 'react-router-dom';

const About = () => {
  const teamMembers = [
    {
      name: 'Ayush',
      role: 'NITRR student',
      image: '',
      description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Doloribus totam cumque odit quasi esse officia.',
    },
    {
      name: 'Ayush',
      role: 'NITRR student',
      image: '',
      description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Doloribus totam cumque odit quasi esse officia.',
    },
    {
      name: 'Ayush',
      role: 'NITRR student',
      image: '',
      description: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Doloribus totam cumque odit quasi esse officia.',
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <section className="bg-blue-900 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl font-bold mb-6">About Mineral Analyser</h1>
            <p className="text-xl max-w-3xl mx-auto">
              We're dedicated to making mineral identification and analysis accessible
              to everyone through cutting-edge AI technology and expert geological
              knowledge.
            </p>
          </div>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
              <p className="text-gray-600 mb-4">
                At Mineral Analyser, we believe that understanding the composition of
                rocks and minerals should be accessible to everyone, from professional
                geologists to curious enthusiasts.
              </p>
              <p className="text-gray-600 mb-4">
                Our mission is to provide accurate, instant mineral identification
                using advanced artificial intelligence, making geological analysis
                more efficient and accessible than ever before.
              </p>
              <Link
                to="/analyzer"
                className="inline-block bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 transition duration-300"
              >
                Try Our analyzerr
              </Link>
            </div>
            <div className="relative h-96">
              <img
                src='../assets/img12.jpg'
                alt="Geological Analysis"
                className="absolute inset-0 w-full h-full object-cover rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center mb-12">Our Expert Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {teamMembers.map((member) => (
              <div
                key={member.name}
                className="bg-white rounded-lg shadow-lg overflow-hidden"
              >
                <img
                  src={member.image}
                  alt={member.name}
                  className="w-full h-64 object-cover"
                />
                <div className="p-6"> <h3 className="text-lg font-bold">{member.name}</h3>
                  <p className="text-gray-600">{member.role}</p>
                  <p className="text-gray-600">{member.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;