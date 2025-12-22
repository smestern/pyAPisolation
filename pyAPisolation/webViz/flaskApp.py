"""
Flask server for dynamic mode webViz
⚠️ EXPERIMENTAL - This mode is under active development for future data streaming features
"""
from flask import Flask, jsonify, render_template, request, send_from_directory, abort
from .. import loadFile
import os
import logging
from scipy.signal import resample, decimate
import numpy as np

logger = logging.getLogger(__name__)

class tsServer:
    """Time series data server for dynamic webViz mode
    
    This server provides API endpoints to load electrophysiology traces on-demand
    rather than pre-generating static images. Supports CORS for cross-origin requests.
    """
    
    def __init__(self, config, static=False):
        self.config = config
        self.app = Flask(__name__,
                        root_path=config.output_path if config else './',
                        static_url_path='',
                        static_folder='',
                        template_folder='web/templates')
        self.setup_routes()
        self._configure_cors()

    def _configure_cors(self):
        """Configure CORS headers for all routes"""
        @self.app.after_request
        def after_request(response):
            response.headers.add('Access-Control-Allow-Origin', '*')
            response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
            response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
            return response

    def _validate_path(self, foldername, data_id):
        """Validate and sanitize file paths to prevent directory traversal attacks"""
        if not foldername or not data_id:
            return None
            
        # Remove any path traversal attempts
        foldername = os.path.normpath(foldername).replace('..', '')
        data_id = os.path.normpath(data_id).replace('..', '')
        
        # Construct full path
        full_path = os.path.join(foldername, data_id + '.abf')
        
        # Check if file exists
        if not os.path.exists(full_path):
            logger.warning(f"File not found: {full_path}")
            return None
            
        return full_path

    def setup_routes(self):
        @self.app.route('/api/<string:data_id>')
        def get_data(data_id):
            """API endpoint to load ABF trace data
            
            Query parameters:
                foldername: Path to folder containing ABF files
                
            Returns:
                JSON array where first element is time series, subsequent elements are voltage traces
            """
            try:
                foldername = request.args.get('foldername', '')
                
                # Validate and get file path
                full_path = self._validate_path(foldername, data_id)
                if not full_path:
                    logger.error(f"Invalid or missing file: {data_id} in {foldername}")
                    abort(404, description="Trace file not found")
                
                # Load ABF file
                x, y, z = loadFile.loadABF(full_path)
                
                # Decimate to reduce data size for web transfer
                y = decimate(y, 4, axis=1)
                x = decimate(x, 4, axis=1)
                
                # Trim to 2.5 seconds
                idx = np.argmin(np.abs(x-2.5))
                y = y[:, :idx]
                
                # Format as [time, sweep1, sweep2, ...]
                response_data = np.vstack((x[0, :idx], y)).tolist()
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error loading trace {data_id}: {str(e)}")
                abort(500, description=f"Error loading trace: {str(e)}")
        
        @self.app.route('/')
        def index():
            """Serve main HTML page"""
            try:
                return send_from_directory('', path='./index.html')
            except Exception as e:
                logger.error(f"Error serving index: {str(e)}")
                abort(500, description="Error serving page")
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({"error": "Not found", "message": str(error)}), 404
            
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({"error": "Internal server error", "message": str(error)}), 500

    def run(self, host='127.0.0.1', port=8000, debug=False):
        """Run Flask development server
        
        For production, use gunicorn instead:
            gunicorn -w 4 -b 0.0.0.0:8000 pyAPisolation.webViz.flaskApp:app
        """
        logger.info(f"Starting Flask server on {host}:{port}")
        logger.warning("⚠️ EXPERIMENTAL: Dynamic mode is under development")
        logger.warning("For production use, deploy with gunicorn")
        self.app.run(host=host, port=port, debug=debug)

# Create app instance for gunicorn
app = Flask(__name__)

if __name__ == '__main__':
    from .webVizConfig import webVizConfig
    config = webVizConfig()
    server = tsServer(config=config, static=False)
    server.run()