import { useState, useRef, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import Lenis from 'lenis';
import 'lenis/dist/lenis.css';
import logoImg from './assets/logo4.jpeg';
import CanvasVideo from './components/CanvasVideo';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'video'
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [isAppLoading, setIsAppLoading] = useState(true);
  const fileInputRef = useRef(null);

  useEffect(() => {
    // Initial App Loader
    const loadingTimer = setTimeout(() => {
      setIsAppLoading(false);
    }, 2000);

    const lenis = new Lenis({
      duration: 1.2, // Slightly slower for more premium feel
      smoothWheel: true,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)) // Apple-style exponential ease
    });

    let hasAutoScrolled = false;

    lenis.on('scroll', (e) => {
      ScrollTrigger.update();

      if (e.scroll < 10 && e.velocity > 0 && !hasAutoScrolled) {
        hasAutoScrolled = true;
        lenis.scrollTo('bottom', {
          duration: 4, 
          easing: (t) => 1 - Math.pow(1 - t, 3) 
        });
      }

      if (e.scroll <= 0) {
        hasAutoScrolled = false;
      }
    });

    gsap.ticker.add((time) => {
      lenis.raf(time * 1000);
    });
    gsap.ticker.lagSmoothing(0);

    return () => {
      lenis.destroy();
      clearTimeout(loadingTimer);
    };
  }, []);

  function handleFile(file) {
    if (!file) return;

    // Validate that the file is an image before processing or sending it to the backend
    if (!file.type.startsWith('image/')) {
      alert("Invalid file type. Please upload an image file (e.g., JPEG, PNG).");
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
    setFileType('image');
  }

  function handleFileChange(event) {
    handleFile(event.target.files[0]);
  }

  function handleDrag(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }

  function handleBrowseClick() {
    fileInputRef.current?.click();
  }

  function clearSelection() {
    setSelectedFile(null);
    setPreviewUrl(null);
    setFileType(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleAnalyze() {
    if (!selectedFile) return;

    setLoading(true);
    setResult(null);
    setStatusMessage("Connecting to secure server...");

    // Connect to Python Backend
    const ws = new WebSocket('ws://localhost:8000/ws/analyze');

    ws.onopen = () => {
      setStatusMessage("Uploading media file...");
      const reader = new FileReader();
      reader.onload = (e) => {
        ws.send(e.target.result); // Send binary array buffer
      };
      reader.readAsArrayBuffer(selectedFile);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'received') {
        // Backend received it, show notification!
        setStatusMessage(data.message);
      } else if (data.status === 'complete') {
        // Backend finished processing
        setResult(data);
        setLoading(false);
        setStatusMessage(null);
        ws.close();
      } else if (data.status === 'error') {
        alert(data.message);
        setLoading(false);
        setStatusMessage(null);
        ws.close();
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      alert('Failed to connect to backend server. Make sure Python is running.');
      setLoading(false);
      setStatusMessage(null);
    };
  }

  return (
    <div className="app-container">
      {/* Initial 2-second Loading Screen */}
      {isAppLoading && (
        <div className="initial-loader">
          <h1 className="blinking-argus">ARGUS</h1>
        </div>
      )}

      {/* High Performance Canvas Scrubbing Video */}
      <CanvasVideo />

      {/* Floating Pill Navbar */}
      <nav className="navbar-pill-wrapper">
        <div className="navbar-pill">
          
          <div className="brand">
            <img 
              src={logoImg} 
              alt="Argus Logo"
              className="logo-img"
            />
          </div>

          <div className="nav-links center-links">
            <a href="#how-it-works">How It Works</a>
            <a href="#technology">Technology</a>
            <a href="#api">API</a>
          </div>
          
          <div className="nav-action">
            <a href="#contact" className="btn-pill-outline">Contact</a>
          </div>

        </div>
      </nav>

      <div className="sections-wrapper">
        
        {/* Minimalist Hero Section */}
        <section className="scroll-section hero-section">
          <div className="hero-content">
            <h2 className="minimal-title">The Future of<br/>Digital Trust.</h2>
            <p className="minimal-subtitle">
              Advanced neural networks built to seamlessly detect artificial manipulations.
            </p>
            <button className="btn-primary" onClick={() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth'})}>Detect Media</button>
          </div>
        </section>

        {/* About Section - Minimalist */}
        <section id="how-it-works" className="scroll-section about-section">
          <div className="about-content">
            <h3 className="minimal-section-title">Precision Forensics.</h3>
            <div className="about-grid">
              <div className="about-card-minimal">
                <h4>Pixel-Level Analysis</h4>
                <p>Scans individual pixels for microscopic inconsistencies introduced by generative AI models.</p>
              </div>
              <div className="about-card-minimal">
                <h4>Temporal Consistency</h4>
                <p>Analyzes frame-to-frame variations that expose artificial manipulation in videos.</p>
              </div>
              <div className="about-card-minimal">
                <h4>Metadata Forensics</h4>
                <p>Investigates deep file structures and metadata signatures hidden within synthetic media.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Detector Section (Bottom) */}
        <section className="scroll-section detector-section">
          
          <main className="main-content">
            <div className="glass-panel-minimal detector-container">
              
              <div className="trust-badges-minimal">
                <span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px', verticalAlign: 'text-bottom'}}><polyline points="20 6 9 17 4 12"></polyline></svg>
                  99.4% Accuracy
                </span>
                <span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px', verticalAlign: 'text-bottom'}}><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                  Secure & Private
                </span>
                <span>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '6px', verticalAlign: 'text-bottom'}}><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                  Real-time
                </span>
              </div>

              {!previewUrl && !result && !loading && (
                <div 
                  className={`upload-area-minimal ${dragActive ? "drag-active" : ""}`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                  onClick={handleBrowseClick}
                >
                  <div className="upload-icon-minimal">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                  </div>
                  <p>Drag & drop or click to upload image</p>
                  <input 
                    ref={fileInputRef}
                    type="file" 
                    accept="image/*" 
                    className="file-input" 
                    onChange={handleFileChange} 
                  />
                </div>
              )}

              {previewUrl && !result && !loading && (
                <div className="preview-container">
                  {fileType === 'video' ? (
                    <video src={previewUrl} controls className="media-preview" />
                  ) : (
                    <img src={previewUrl} alt="preview" className="media-preview" />
                  )}

                  <div className="action-buttons">
                    <button className="btn-secondary" onClick={clearSelection} disabled={loading}>
                      Clear
                    </button>
                    <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
                      Analyze
                    </button>
                  </div>
                </div>
              )}

              {loading && (
                <div className="loader-wrapper">
                  <div className="loader-minimal"></div>
                  <p className="loader-text">{statusMessage || 'Analyzing pixels...'}</p>
                </div>
              )}

              {result && !loading && (
                <div className="result-container-minimal">
                  <h3 className="result-title">
                    <span className="result-authentic">
                      {result.prediction === 'apple' ? 'Apple' : 'Tomato'} detected
                    </span>
                  </h3>
                  
                  <div className="confidence-bar-minimal-bg">
                    <div 
                      className="confidence-bar-minimal-fill" 
                      style={{ 
                        width: `${result.confidence * 100}%`,
                        background: result.prediction === 'apple' ? '#34c759' : '#ff3b30'
                      }}
                    ></div>
                  </div>
                  <p className="confidence-text-minimal">
                    Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                  </p>

                  <div style={{ marginTop: '1.5rem' }}>
                    <button className="btn-secondary" onClick={clearSelection}>
                      Scan Another
                    </button>
                  </div>
                </div>
              )}

            </div>
          </main>
        </section>

      </div>
    </div>
  );
}

export default App;