import os
import sys
from http.server import HTTPServer, CGIHTTPRequestHandler
os.chdir("./pyAPisolation/")
os.chdir("./web_viz")
sys.path.append('..')
sys.path.append('')
# Create server object listening the port 80
server_object = HTTPServer(server_address=('127.0.0.1', 800), RequestHandlerClass=CGIHTTPRequestHandler)
# Start the web server
server_object.serve_forever()
