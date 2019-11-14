# pyAPisolation
pyAPisolation isolates and extracts action potentials from ABF files (recorded in current clamp mode). The package is able to output feature arrays and/or raw current traces from the file(s). The module piggy backs off of some of the excellent work found in the pyABF package.
### Installation
To install run
  `pip install git+https://github.com/smestern/pyAPisolation`
### Basic use
To run simple analysis on your files. navigate to the /bin folder and run:
  
  `python run_APisolation.py`
  
This will prompt you to select your files for analysis. The script will create an output folder for your traces.

### Advanced use
If you only want to use certain extraction features you can import the extraction module directly

  `import pyAPisolation as apis`

from here you can choose to use specific features. For example, if you want to extract only the average threshold

   ` apis.nuactionpotential.thresholdavg(abf_file,filter)`
   
 **nuactionpotential** provides modules for action potential extraction, **abfderivative** provides support function to return the derivative / integral / double derivative of an abf file. 
 
    ` slopex, slopey = apis.abfderivative.derivative(abf,sweepNumber) `
