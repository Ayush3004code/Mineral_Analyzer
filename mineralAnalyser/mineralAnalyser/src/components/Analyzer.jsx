import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { FiUpload, FiX, FiImage, FiVideo } from 'react-icons/fi';
import axios from 'axios';

const Analyzer = () => {
  const [files, setFiles] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [responseImages, setResponseImages] = useState(null);
  const [predictedClass, setPredictedClass] = useState('');

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => {
      const isVideo = file.type.startsWith('video/');
      return Object.assign(file, {
        preview: URL.createObjectURL(file),
        isVideo
      });
    });
    setFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif'],
      'video/mp4': ['.mp4']
    },
    multiple: true
  });

  const removeFile = (fileToRemove) => {
    setFiles(files.filter(file => file !== fileToRemove));
    URL.revokeObjectURL(fileToRemove.preview);
  };

  const handleAnalyze = async () => {
    if (files.length === 0) return;

    setIsAnalyzing(true);
    setProgress(0);

    const formData = new FormData();
    formData.append('file', files[0]);

    try {
      const response = await axios.post('http://127.0.0.1:5000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const { loaded, total } = progressEvent;
          setProgress(Math.round((loaded * 100) / total));
        }
      });
      console.log(response)
      setResponseImages({
        segmentedImage: response.data.segmented_image,
        boundaryImage: response.data.boundary_image,
        barChart: response.data.bar_chart,
        pieChart: response.data.pie_chart,
      });
      setPredictedClass(response.data.predicted_class);
    } catch (error) {
      console.error("There was an error uploading the file!", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getFileSize = (size) => {
    const mb = size / (1024 * 1024);
    return mb.toFixed(2) + ' MB';
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Mineral Analyzer</h1>
          <p className="mt-2 text-gray-600">Upload images or videos of minerals for instant analysis</p>
        </div>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
          }`}
        >
          <input {...getInputProps()} />
          <FiUpload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-gray-600">
            {isDragActive ? 'Drop the files here ...' : 'Drag & drop files here, or click to select files'}
          </p>
          <p className="text-sm text-gray-500 mt-1">Supported formats: JPEG, PNG, GIF, MP4</p>
        </div>

        {files.length > 0 && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Selected Files ({files.length})
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {files.map((file, index) => (
                <div key={index} className="relative rounded-lg overflow-hidden">
                  {file.isVideo ? (
                    <video src={file.preview} className="w-full h-full object-cover" controls />
                  ) : (
                    <img src={file.preview} alt={file.name} className="w-full h-full object-cover" />
                  )}
                  <button
                    onClick={() => removeFile(file)}
                    className="absolute top-2 right-2 text-gray-500 hover:text-gray-800 transition-colors"
                  >
                    <FiX />
                  </button>
                  <p className="text-sm text-gray-500 mt-1">{file.name} ({getFileSize(file.size)})</p>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex justify-center mt-8">
          {!isAnalyzing ? (
            <button
              onClick={handleAnalyze}
              className={`${
                files.length === 0 ? 'bg-slate-500' : 'bg-blue-500 hover:bg-blue-700'
              } text-white font-bold py-3 px-8 rounded-lg transition-all duration-300 transform hover:scale-105 hover:shadow-lg`}
              disabled={files.length === 0}
            >
              Analyze Files
            </button>
          ) : (
            <div className="w-64 relative">
              <div className="w-full bg-gray-200 h-12 rounded-lg overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-300 relative"
                  style={{ width: `${progress}%` }}
                />
                <p className="text-sm text-gray-500 mt-1">{progress}% Analyzed</p>
              </div>
            </div>
          )}
        </div>
        {/* Display Response Images */}
{responseImages && (
  <div className="mt-8">
    <h2 className="text-lg font-semibold text-gray-900 mb-4">Analysis Results</h2>
    {/* Predicted Class Display */}
    <h3 className="text-md font-semibold text-gray-800 mb-2">
       Predicted Class: {predictedClass}  {/* Use predictedClass state here */}
    </h3>
    <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-1">Segmented Image</h3>
        <img
          src={`http://127.0.0.1:5000/${responseImages.segmentedImage}`}
          alt="Segmented Image"
          className="w-full h-auto rounded-lg"
        />
      </div>
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-1">Boundary Image</h3>
        <img
          src={`http://127.0.0.1:5000/${responseImages.boundaryImage}`}
          alt="Boundary Image"
          className="w-full h-auto rounded-lg"
        />
      </div>
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-1">Bar Chart</h3>
        <img
          src={`http://127.0.0.1:5000/${responseImages.barChart}`}
          alt="Bar Chart"
          className="w-full h-auto rounded-lg"
        />
      </div>
      <div>
        <h3 className="text-sm font-medium text-gray-700 mb-1">Pie Chart</h3>
        <img
          src={`http://127.0.0.1:5000/${responseImages.pieChart}`}
          alt="Pie Chart"
          className="w-full h-auto rounded-lg"
        />
      </div>
    </div>
  </div>
)}

      </div>
    </div>
  );
};

export default Analyzer;
