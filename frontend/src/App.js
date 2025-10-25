import './App.css';
import React, { useState } from 'react';

function ModelCard({ item }) {
  if (item.error) {
    return (
      <div className="result-card">
        <h3>{item.model}</h3>
        <div className="error">Error: {item.error}</div>
      </div>
    );
  }

  return (
    <div className="result-card">
      <h3>{item.model}</h3>
      <div className="result-row"><strong>Predicted:</strong> {item.label}</div>
      <div className="result-row"><strong>Confidence:</strong> {(item.confidence * 100).toFixed(2)}%</div>
    </div>
  );
}

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  function onFileChange(e) {
    setError(null);
    const f = e.target.files[0];
    if (!f) return;
    if (!f.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }
    setFile(f);
    const url = URL.createObjectURL(f);
    setPreview(url);
    setResults([]);
  }

  async function onPredict() {
    if (!file) {
      setError('No file selected');
      return;
    }
    setPredicting(true);
    setError(null);
    setResults([]);

    try {
      const form = new FormData();
      form.append('file', file);

      const resp = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: form,
      });

      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server error: ${resp.status} ${text}`);
      }

      const data = await resp.json();
      if (data.results) setResults(data.results);
      else setError('No results from server');
    } catch (e) {
      setError(String(e));
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="app-root">
      <header className="header">
        <h1>Skin Lesion Detection</h1>
      </header>

      <main className="main">
        <section className="upload">
          <label className="upload-box" htmlFor="file-input">
            <div className="upload-text">Click to select an image or drag here</div>
            <input id="file-input" type="file" accept="image/*" onChange={onFileChange} />
          </label>

          <div className="controls">
            <button className="predict-btn" onClick={onPredict} disabled={predicting || !file}>
              {predicting ? 'Predicting...' : 'Predict'}
            </button>
          </div>

          {error && <div className="error-banner">{error}</div>}
        </section>

        <section className="preview-results">
          <div className="preview">
            <h2>Uploaded Image</h2>
            {preview ? (
              <img src={preview} alt="uploaded preview" />
            ) : (
              <div className="empty-preview">No image selected</div>
            )}
          </div>

          <div className="results">
            <h2>Model Predictions</h2>
            {predicting && <div className="spinner">Predicting — please wait</div>}
            {!predicting && results.length === 0 && <div className="empty-results">No predictions yet</div>}

            <div className="results-grid">
              {results.map((r, i) => (
                <ModelCard key={i} item={r} />
              ))}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
