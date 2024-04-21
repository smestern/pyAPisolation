from flask import Flask, jsonify, render_template, request, send_from_directory
from .. import loadABF
import os
from scipy.signal import resample, decimate
import numpy as np

class traceserver:
    def __init__(self, config, static=False):
        self.app = Flask(__name__,
                         root_path=config.output_path,
                        static_url_path='',
                         static_folder='',
                        template_folder='web/templates')
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/api/<string:data_id>')
        def get_data(data_id):
            foldername = request.args.get('foldername')
            x, y, z = loadABF.loadABF(os.path.join(foldername, data_id+'.abf'))
            y = decimate(y, 4, axis=1)
            x = decimate(x, 4, axis=1)
            idx = np.argmin(np.abs(x-2.5))
            y = y[:, :idx]
            y = np.vstack((x[0, :idx], y)).tolist()
            return self._corsify_actual_response(jsonify(y ))
        
        @self.app.route('/')
        def index():
            return send_from_directory('',path='./output.html')
        
    def _corsify_actual_response(self, response):
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    def run(self):
        self.app.run()

if __name__ == '__main__':
    traceserver(config=None, static=False).run()