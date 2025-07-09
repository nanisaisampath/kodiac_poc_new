
import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [viewports, setViewports] = useState([
    { id: 1, title: 'Viewport 1 - Left Eye', file: null, currentFrame: 0, totalFrames: 0, dicomFilePath: '', isLeftEye: false },
    { id: 2, title: 'Viewport 2 - Right Eye', file: null, currentFrame: 0, totalFrames: 0, dicomFilePath: '', isRightEye: false }
  ]);
  const [bindSliders, setBindSliders] = useState(false);
  const [isE2ELoaded, setIsE2ELoaded] = useState(false);
  const [focusedEye, setFocusedEye] = useState(null);
  const [testMessage, setTestMessage] = useState('');
  const [s3Files, setS3Files] = useState([]);
  const [selectedS3File, setSelectedS3File] = useState(null);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    loadS3Files();
  }, []);

  const loadS3Files = async () => {
    try {
      const response = await fetch('/api/s3-flat-list');
      if (response.ok) {
        const files = await response.json();
        setS3Files(files);
      } else {
        console.error('Failed to load S3 files');
      }
    } catch (error) {
      console.error('Error loading S3 files:', error);
    }
  };

  const handleS3FileRightClick = (event, file) => {
    event.preventDefault();
    if (file.key.toLowerCase().endsWith('.e2e')) {
      setSelectedS3File(file);
      setContextMenuPosition({ x: event.clientX, y: event.clientY });
      setShowContextMenu(true);
    }
  };

  const handleLoadE2E = async () => {
    if (!selectedS3File) return;

    try {
      setShowContextMenu(false);
      
      // Download and process the E2E file from S3
      const response = await fetch(`/api/download_dicom_from_s3?path=${encodeURIComponent(selectedS3File.key)}`);
      
      if (response.ok) {
        const data = await response.json();
        
        // Update both viewports for left and right eye
        setViewports([
          { 
            id: 1, 
            title: 'Viewport 1 - Left Eye',
            totalFrames: data.number_of_frames, 
            dicomFilePath: data.dicom_file_path,
            isLeftEye: true,
            currentFrame: 0,
            file: selectedS3File
          },
          { 
            id: 2, 
            title: 'Viewport 2 - Right Eye',
            totalFrames: data.number_of_frames, 
            dicomFilePath: data.dicom_file_path,
            isRightEye: true,
            currentFrame: 0,
            file: selectedS3File
          }
        ]);
        
        // Display both left and right eye frames
        await displayE2EFrame(1, 0, data.dicom_file_path, 'left');
        await displayE2EFrame(2, 0, data.dicom_file_path, 'right');
        
        setIsE2ELoaded(true);
      } else {
        const errorData = await response.json();
        alert('Error: ' + errorData.detail);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while loading the E2E file.');
    }
  };

  const displayE2EFrame = async (viewportNumber, frame, dicomFilePath, eye) => {
    try {
      const response = await fetch(
        `/api/view_e2e_eye?frame=${frame}&dicom_file_path=${encodeURIComponent(dicomFilePath)}&eye=${eye}`
      );
      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
  
        // Update the viewport's imageUrl and currentFrame
        setViewports((prevViewports) =>
          prevViewports.map((viewport) =>
            viewport.id === viewportNumber
              ? { ...viewport, currentFrame: frame, imageUrl: imageUrl }
              : viewport
          )
        );
      } else {
        const errorData = await response.json();
        alert(`Error: ${errorData.detail}`);
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while displaying the E2E frame.');
    }
  };

  const testBackendConnection = async () => {
    try {
      console.log('Attempting to connect to /api/test');
      const response = await fetch('/api/test');
      console.log('Response received:', response);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      console.log('Response data:', data);
      setTestMessage(data.message);
    } catch (error) {
      console.error('Error:', error);
      setTestMessage(`Connection failed: ${error.message}`);
    }
  };

  const handleSliderChange = (viewportNumber, frame) => {
    const viewport = viewports.find((v) => v.id === viewportNumber);
    if (!viewport) return;
  
    const eye = viewport.isLeftEye ? 'left' : 'right';
    displayE2EFrame(viewportNumber, frame, viewport.dicomFilePath, eye);
  
    if (bindSliders) {
      const otherViewportNumber = viewportNumber === 1 ? 2 : 1;
      const otherViewport = viewports.find((v) => v.id === otherViewportNumber);
      if (otherViewport && otherViewport.totalFrames === viewport.totalFrames) {
        const otherEye = otherViewport.isLeftEye ? 'left' : 'right';
        displayE2EFrame(otherViewportNumber, frame, otherViewport.dicomFilePath, otherEye);
      }
    }
  };

  const handleFocusEye = (eye) => {
    setFocusedEye(eye);
    // You can add logic here to highlight or emphasize the focused eye viewport
  };

  const handleE2EToDicomConvert = async () => {
    // This functionality can be removed or kept for conversion purposes
    alert('E2E to DICOM conversion functionality can be accessed through the tools menu.');
  };

  const handleDicomMetadataExtract = async () => {
    // This functionality can be removed or kept for metadata extraction
    alert('DICOM metadata extraction functionality can be accessed through the tools menu.');
  };

  // Close context menu when clicking elsewhere
  useEffect(() => {
    const handleClickOutside = () => {
      setShowContextMenu(false);
    };

    if (showContextMenu) {
      document.addEventListener('click', handleClickOutside);
    }

    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [showContextMenu]);

  return (
    <div className="App">
      <div className="menu-bar">
        <nav>
          <a href="#">File</a>
          <a href="#">Edit</a>
          <div className="dropdown">
            <span className="dropbtn">Tools</span>
            <div className="dropdown-content">
              <a href="#">Lock Scroll</a>
              <a href="#" onClick={handleE2EToDicomConvert}>E2E to DICOM Converter</a>
              <a href="#" onClick={handleDicomMetadataExtract}>DICOM Metadata & Pixel Extractor</a>
              <a href="#">Other Tool 2</a>
            </div>
          </div>
          <a href="#">Help</a>
        </nav>
      </div>
      <header>Retinal Image Viewer and File Processor</header>
  
      <div className="main-content">
        <div className="test-connection">
          <button onClick={testBackendConnection}>Test Backend Connection</button>
          {testMessage && <p>{testMessage}</p>}
        </div>

        {/* S3 File Browser */}
        <div className="s3-browser">
          <h3>S3 Files Browser</h3>
          <button onClick={loadS3Files}>Refresh S3 Files</button>
          <div className="s3-file-list">
            {s3Files.map((file, index) => (
              <div
                key={index}
                className={`s3-file-item ${file.key.toLowerCase().endsWith('.e2e') ? 'e2e-file' : ''}`}
                onContextMenu={(e) => handleS3FileRightClick(e, file)}
                style={{
                  padding: '8px',
                  margin: '2px 0',
                  backgroundColor: file.key.toLowerCase().endsWith('.e2e') ? '#e3f2fd' : '#f5f5f5',
                  cursor: file.key.toLowerCase().endsWith('.e2e') ? 'context-menu' : 'default',
                  borderRadius: '4px'
                }}
              >
                <div>{file.key}</div>
                <div style={{ fontSize: '0.8em', color: '#666' }}>
                  {(file.size / (1024 * 1024)).toFixed(2)} MB - {file.last_modified}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Context Menu */}
        {showContextMenu && selectedS3File && (
          <div
            className="context-menu"
            style={{
              position: 'fixed',
              left: contextMenuPosition.x,
              top: contextMenuPosition.y,
              backgroundColor: 'white',
              border: '1px solid #ccc',
              borderRadius: '4px',
              padding: '8px 0',
              boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
              zIndex: 1000
            }}
          >
            <div
              className="context-menu-item"
              onClick={handleLoadE2E}
              style={{
                padding: '8px 16px',
                cursor: 'pointer',
                backgroundColor: 'white'
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#f0f0f0'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
            >
              Load E2E
            </div>
          </div>
        )}

        {/* E2E Viewports */}
        {isE2ELoaded && (
          <>
            {/* Eye Focus Controls */}
            <div className="eye-focus-controls" style={{ textAlign: 'center', margin: '20px 0' }}>
              <button
                onClick={() => handleFocusEye('left')}
                style={{
                  margin: '0 10px',
                  padding: '10px 20px',
                  backgroundColor: focusedEye === 'left' ? '#4CAF50' : '#f0f0f0',
                  color: focusedEye === 'left' ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Focus Left Eye
              </button>
              <button
                onClick={() => handleFocusEye('right')}
                style={{
                  margin: '0 10px',
                  padding: '10px 20px',
                  backgroundColor: focusedEye === 'right' ? '#4CAF50' : '#f0f0f0',
                  color: focusedEye === 'right' ? 'white' : 'black',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Focus Right Eye
              </button>
            </div>

            {/* Viewports */}
            <div className="viewports-container" style={{ display: 'flex', gap: '20px' }}>
              {viewports.map(viewport => (
                <div key={viewport.id} className="viewport-container" style={{ flex: 1 }}>
                  <h2 id={`viewportTitle${viewport.id}`}>{viewport.title}</h2>
                  {viewport.imageUrl ? (
                    <>
                      <img 
                        id={`viewportImage${viewport.id}`} 
                        className="viewport-image" 
                        src={viewport.imageUrl} 
                        alt={`E2E ${viewport.title}`}
                        style={{
                          opacity: focusedEye && 
                            ((focusedEye === 'left' && !viewport.isLeftEye) || 
                             (focusedEye === 'right' && !viewport.isRightEye)) ? 0.3 : 1,
                          transition: 'opacity 0.3s'
                        }}
                      />
                      <div className="slider-section">
                        <label htmlFor={`frameSlider${viewport.id}`}>Select Frame</label>
                        <input
                          type="range"
                          id={`frameSlider${viewport.id}`}
                          min="0"
                          max={viewport.totalFrames - 1}
                          value={viewport.currentFrame}
                          onChange={(e) => handleSliderChange(viewport.id, parseInt(e.target.value))}
                        />
                        <p>Current Frame: {viewport.currentFrame + 1} of {viewport.totalFrames}</p>
                      </div>
                    </>
                  ) : (
                    <div className="loading-placeholder">Loading {viewport.title}...</div>
                  )}
                </div>
              ))}
            </div>

            <div id="bindSlidersContainer">
              <input 
                type="checkbox" 
                id="bindSliders" 
                checked={bindSliders}
                onChange={(e) => setBindSliders(e.target.checked)}
              />
              <label htmlFor="bindSliders">Bind Sliders</label>
            </div>
          </>
        )}
      </div>
  
      <footer>
        © 2024 Kodiak Sciences Inc - All Rights Reserved.
      </footer>
    </div>
  );
}

export default App;
