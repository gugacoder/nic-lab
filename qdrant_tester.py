
#!/usr/bin/env python3
"""
QDrant Data Quality Testing Application
=======================================

Flask application for testing data quality in Qdrant vector database.
Integrates sentence-transformers for embeddings and provides web interface.

Requirements:
pip install flask sentence-transformers requests

Usage:
python qdrant_tester.py

Then visit: https://localhost:5000
"""

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings

# Suppress sentence-transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import requests
    from flask import Flask, render_template_string, request, jsonify, session
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Error: Missing required packages. Please run:")
    print("pip install flask sentence-transformers requests")
    sys.exit(1)

app = Flask(__name__)
app.secret_key = 'qdrant-testing-app-secret-key-2024'

# Configuration file path
CONFIG_FILE = 'qdrant_config.json'

# Global embedding model
embedding_model = None

# HTML Template with embedded CSS and JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QDrant Data Quality Testing</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .form-container {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #374151;
            font-size: 14px;
        }
        
        .form-group input,
        .form-group select,
        .form-group textarea {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
            background: #f9fafb;
        }
        
        .form-group input:focus,
        .form-group select:focus,
        .form-group textarea:focus {
            outline: none;
            border-color: #4f46e5;
            background: white;
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        
        .form-row {
            display: flex;
            gap: 20px;
        }
        
        .form-row .form-group {
            flex: 1;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(79, 70, 229, 0.3);
        }
        
        .btn-secondary {
            background: #6b7280;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #4b5563;
            transform: translateY(-2px);
        }
        
        .btn-group {
            display: flex;
            gap: 15px;
            margin-top: 20px;
        }
        
        .divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
            margin: 30px 0;
        }
        
        .output-section {
            background: #f8fafc;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .output-section h3 {
            color: #374151;
            margin-bottom: 15px;
            font-size: 18px;
        }
        
        .results {
            background: white;
            border-radius: 6px;
            padding: 20px;
            border: 1px solid #e5e7eb;
            max-height: 500px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            white-space: pre-wrap;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }
        
        .loading .spinner {
            border: 3px solid #f3f4f6;
            border-top: 3px solid #4f46e5;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: inline-block;
            animation: spin 1s linear infinite;
            margin-bottom: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #dc2626;
            padding: 15px;
            border-radius: 6px;
            margin-top: 10px;
        }
        
        .success {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #166534;
            padding: 15px;
            border-radius: 6px;
            margin-top: 10px;
        }
        
        .collection-group {
            display: flex;
            gap: 10px;
            align-items: end;
        }
        
        .collection-group .form-group {
            flex: 1;
            margin-bottom: 0;
        }
        
        textarea {
            resize: vertical;
            min-height: 100px;
        }
        
        @media (max-width: 768px) {
            .form-row {
                flex-direction: column;
            }
            
            .btn-group {
                flex-direction: column;
            }
            
            .collection-group {
                flex-direction: column;
                align-items: stretch;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>QDrant Data Quality Testing</h1>
            <p>Vector database quality analysis and search testing</p>
        </div>
        
        <div class="form-container">
            <form id="qdrantForm">
                <div class="form-row">
                    <div class="form-group">
                        <label for="host">Host:</label>
                        <input type="url" id="host" name="host" placeholder="https://your-qdrant-host:6333" required>
                    </div>
                    <div class="form-group">
                        <label for="api_key">API Key:</label>
                        <input type="password" id="api_key" name="api_key" placeholder="Optional API Key">
                    </div>
                </div>
                
                <div class="collection-group">
                    <div class="form-group">
                        <label for="collection">Collection:</label>
                        <select id="collection" name="collection" required>
                            <option value="">Select a collection...</option>
                        </select>
                    </div>
                    <button type="button" class="btn btn-secondary" onclick="fetchCollections()">
                        Buscar Collections
                    </button>
                </div>
                
                <div class="form-row">
                    <div class="form-group">
                        <label for="embedding_algorithm">Embedding Algorithm:</label>
                        <select id="embedding_algorithm" name="embedding_algorithm">
                            <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (384 dims)</option>
                            <option value="all-mpnet-base-v2">all-mpnet-base-v2 (768 dims)</option>
                            <option value="all-MiniLM-L12-v2">all-MiniLM-L12-v2 (384 dims)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="vector_field">Vector Field Name:</label>
                        <input type="text" id="vector_field" name="vector_field" value="vector" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="max_results">Max Results:</label>
                    <input type="number" id="max_results" name="max_results" value="10" min="1" max="100" required>
                </div>
                
                <div class="form-group">
                    <label for="query_input">Query Input:</label>
                    <textarea id="query_input" name="query_input" placeholder="Enter your search query here..." required></textarea>
                </div>
                
                <div class="divider"></div>
                
                <div class="btn-group">
                    <button type="submit" class="btn btn-primary">Submit Query</button>
                    <button type="button" class="btn btn-secondary" onclick="clearResults()">Clear Results</button>
                </div>
            </form>
            
            <div class="output-section">
                <h3>Output:</h3>
                <div id="results" class="results">
                    Ready to process queries. Fill in the form above and click "Submit Query".
                </div>
            </div>
        </div>
    </div>

    <script>
        // Load saved configuration on page load
        window.onload = function() {
            loadConfiguration();
        };
        
        // Save configuration whenever form changes
        document.getElementById('qdrantForm').addEventListener('input', saveConfiguration);
        
        function saveConfiguration() {
            const config = {
                host: document.getElementById('host').value,
                api_key: document.getElementById('api_key').value,
                embedding_algorithm: document.getElementById('embedding_algorithm').value,
                vector_field: document.getElementById('vector_field').value,
                max_results: document.getElementById('max_results').value
            };
            localStorage.setItem('qdrant_config', JSON.stringify(config));
        }
        
        function loadConfiguration() {
            const saved = localStorage.getItem('qdrant_config');
            if (saved) {
                const config = JSON.parse(saved);
                document.getElementById('host').value = config.host || '';
                document.getElementById('api_key').value = config.api_key || '';
                document.getElementById('embedding_algorithm').value = config.embedding_algorithm || 'all-MiniLM-L6-v2';
                document.getElementById('vector_field').value = config.vector_field || 'vector';
                document.getElementById('max_results').value = config.max_results || '10';
            }
        }
        
        async function fetchCollections() {
            const host = document.getElementById('host').value;
            const api_key = document.getElementById('api_key').value;
            
            if (!host) {
                showError('Please enter a host URL first.');
                return;
            }
            
            showLoading('Fetching collections...');
            
            try {
                const response = await fetch('/api/collections', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        host: host,
                        api_key: api_key
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    const select = document.getElementById('collection');
                    select.innerHTML = '<option value="">Select a collection...</option>';
                    
                    data.collections.forEach(collection => {
                        const option = document.createElement('option');
                        option.value = collection;
                        option.textContent = collection;
                        select.appendChild(option);
                    });
                    
                    showSuccess(`Found ${data.collections.length} collections.`);
                } else {
                    showError('Error: ' + data.error);
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        }
        
        document.getElementById('qdrantForm').onsubmit = async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());
            
            if (!data.host || !data.collection || !data.query_input) {
                showError('Please fill in all required fields.');
                return;
            }
            
            showLoading('Processing query and searching...');
            saveConfiguration();
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('results').textContent = result.markdown;
                } else {
                    showError('Search error: ' + result.error);
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        };
        
        function showLoading(message) {
            document.getElementById('results').innerHTML = 
                `<div class="loading">
                    <div class="spinner"></div>
                    <div>${message}</div>
                </div>`;
        }
        
        function showError(message) {
            document.getElementById('results').innerHTML = 
                `<div class="error">${message}</div>`;
        }
        
        function showSuccess(message) {
            document.getElementById('results').innerHTML = 
                `<div class="success">${message}</div>`;
        }
        
        function clearResults() {
            document.getElementById('results').textContent = 
                'Ready to process queries. Fill in the form above and click "Submit Query".';
        }
    </script>
</body>
</html>
"""

class QdrantTester:
    """Main class for Qdrant testing functionality"""
    
    def __init__(self):
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            self.config = {}
    
    def save_config(self, config: Dict):
        """Save configuration to file"""
        try:
            self.config.update(config)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def get_embedding_model(self, model_name: str) -> SentenceTransformer:
        """Load embedding model (cached)"""
        global embedding_model
        
        if embedding_model is None or embedding_model._model_name != model_name:
            try:
                print(f"Loading embedding model: {model_name}")
                embedding_model = SentenceTransformer(model_name)
                embedding_model._model_name = model_name
            except Exception as e:
                raise Exception(f"Failed to load embedding model '{model_name}': {str(e)}")
        
        return embedding_model
    
    def fetch_collections(self, host: str, api_key: Optional[str] = None) -> List[str]:
        """Fetch collections from Qdrant"""
        try:
            # Clean host URL
            host = host.rstrip('/')
            url = f"{host}/collections"
            
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['api-key'] = api_key
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'result' in data and 'collections' in data['result']:
                collections = [col['name'] for col in data['result']['collections']]
                return sorted(collections)
            else:
                raise Exception("Unexpected response format from Qdrant")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection error: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from Qdrant")
        except Exception as e:
            raise Exception(f"Error fetching collections: {str(e)}")
    
    def search_vectors(self, host: str, collection: str, query_vector: List[float], 
                      max_results: int, api_key: Optional[str] = None, 
                      vector_field: str = "vector") -> List[Dict]:
        """Search vectors in Qdrant collection"""
        try:
            # Clean host URL
            host = host.rstrip('/')
            url = f"{host}/collections/{collection}/points/search"
            
            headers = {'Content-Type': 'application/json'}
            if api_key:
                headers['api-key'] = api_key
            
            # First, try with named vector field
            payload_named = {
                "vector": {
                    "name": vector_field,
                    "vector": query_vector
                },
                "limit": max_results,
                "with_payload": True,
                "with_vector": False
            }
            
            response = requests.post(url, headers=headers, json=payload_named, timeout=30)
            
            # If named vector fails, try with unnamed vector
            if response.status_code == 400:
                payload_unnamed = {
                    "vector": query_vector,
                    "limit": max_results,
                    "with_payload": True,
                    "with_vector": False
                }
                
                response = requests.post(url, headers=headers, json=payload_unnamed, timeout=30)
            
            response.raise_for_status()
            
            data = response.json()
            
            if 'result' in data:
                return data['result']
            else:
                raise Exception("Unexpected response format from Qdrant search")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Search request error: {str(e)}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from Qdrant search")
        except Exception as e:
            raise Exception(f"Error during vector search: {str(e)}")
    
    def format_results_markdown(self, results: List[Dict], query: str) -> str:
        """Format search results as markdown"""
        if not results:
            return f"# Query: {query}\n\nNo results found."
        
        markdown = f"# Query: {query}\n\n"
        markdown += f"Found {len(results)} results:\n\n"
        
        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            payload = result.get('payload', {})
            
            # Extract text content from payload
            text_content = ""
            if 'text' in payload:
                text_content = payload['text']
            elif 'content' in payload:
                text_content = payload['content']
            elif 'document' in payload:
                text_content = payload['document']
            else:
                # Try to find any string value in payload
                for key, value in payload.items():
                    if isinstance(value, str) and len(value) > 10:
                        text_content = value
                        break
            
            if not text_content:
                text_content = "No text content found"
            
            # Limit text content length
            if len(text_content) > 500:
                text_content = text_content[:500] + "..."
            
            markdown += f"---\n"
            markdown += f"#{i} (Score: {score:.4f})\n"
            markdown += f"Match: {text_content}\n"
            markdown += f"Metadata: {{\n"
            
            # Format metadata nicely
            for key, value in payload.items():
                if key not in ['text', 'content', 'document']:
                    if isinstance(value, str):
                        markdown += f'    "{key}": "{value}",\n'
                    else:
                        markdown += f'    "{key}": {json.dumps(value)},\n'
            
            markdown += "}\n"
        
        return markdown

# Initialize tester
tester = QdrantTester()

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/collections', methods=['POST'])
def api_collections():
    """API endpoint to fetch collections"""
    try:
        data = request.get_json()
        host = data.get('host')
        api_key = data.get('api_key')
        
        if not host:
            return jsonify({'success': False, 'error': 'Host URL is required'})
        
        # Save configuration
        tester.save_config({'host': host, 'api_key': api_key})
        
        collections = tester.fetch_collections(host, api_key)
        
        return jsonify({'success': True, 'collections': collections})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint to perform vector search"""
    try:
        data = request.get_json()
        
        # Extract parameters
        host = data.get('host')
        api_key = data.get('api_key')
        collection = data.get('collection')
        embedding_algorithm = data.get('embedding_algorithm', 'all-MiniLM-L6-v2')
        vector_field = data.get('vector_field', 'vector')
        max_results = int(data.get('max_results', 10))
        query_input = data.get('query_input')
        
        # Validate required fields
        if not all([host, collection, query_input]):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Save configuration
        config = {
            'host': host,
            'api_key': api_key,
            'embedding_algorithm': embedding_algorithm,
            'vector_field': vector_field,
            'max_results': max_results
        }
        tester.save_config(config)
        
        # Generate embedding
        model = tester.get_embedding_model(embedding_algorithm)
        query_vector = model.encode(query_input).tolist()
        
        # Perform search
        results = tester.search_vectors(
            host=host,
            collection=collection,
            query_vector=query_vector,
            max_results=max_results,
            api_key=api_key,
            vector_field=vector_field
        )
        
        # Format results
        markdown = tester.format_results_markdown(results, query_input)
        
        return jsonify({'success': True, 'markdown': markdown})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

def main():
    """Main function to run the Flask app"""
    print("=" * 60)
    print("QDrant Data Quality Testing Application")
    print("=" * 60)
    print()
    print("Checking dependencies...")
    
    # Check if sentence-transformers models directory exists
    try:
        import sentence_transformers
        print("✓ sentence-transformers installed")
    except ImportError:
        print("✗ sentence-transformers not found")
        print("Please run: pip install sentence-transformers")
        sys.exit(1)
    
    try:
        import requests
        print("✓ requests installed")
    except ImportError:
        print("✗ requests not found")
        print("Please run: pip install requests")
        sys.exit(1)
    
    print("✓ All dependencies ready")
    print()
    print("Starting Flask application...")
    print("Application will be available at:")
    print("  - HTTP:  http://localhost:5000")
    print("  - HTTPS: https://localhost:5000 (self-signed certificate)")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run with HTTP first for easy access, HTTPS available on port 5001
    try:
        print("Starting HTTP server on port 5000...")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except Exception as e:
        print(f"HTTP failed ({e}), trying HTTPS...")
        # Fallback to HTTPS
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            ssl_context='adhoc',
            threaded=True
        )

if __name__ == '__main__':
    main()
