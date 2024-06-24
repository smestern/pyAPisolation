// This script is used to generate the table and parallel coordinates plot in the web application
function maketrace(row){
        var url = "http://localhost:5000/api/" + row.ID + '?foldername=' + row.foldername // replace with your Flask app endpoint
        Plotly.d3.json(url, function(rows){			
                        data = []
                        var i = 1
                        //pop the first row as it is the time series
                        var time_row = rows.shift(0)

                        rows.forEach(function(row) {
                            var sweepname = 'Sweep ' + i + ': ' + Math.round(row[Object.keys(row)[0]].toString()) + ' pA'
                            delete row[Object.keys(row)[0]]
                            var rowdata =  Object.keys(row).map(function(e) { 
                                                                    return row[e]
                                                                })
                            var trace = {
                                    type: 'scattergl',                    // set the chart type
                                    mode: 'lines',                      // connect points with lines
                                    name: sweepname,
                                    y: rowdata,
                                    x: time_row,
                                    hovertemplate: '%{x} S, %{y} mV',
                                    line: {                             // set the width of the line.
                                        width: 1,
                                        shape: 'spline',
                                        smoothing: 0.005
                                    }
                                        
                                    };
                            data.push(trace);
                            i += 1;
                        });
                
                    

                        var layout = {
                        width: "100%",
                        height: "100%",
                        yaxis: {title: "mV",
                                },       // set the y axis title
                        xaxis: {
                            //dtick: 0.25,
                            zeroline: false,
                            title: "Time (S)"// remove the x-axis grid lines
                            
                        },
                        margin: {                           // update the left, bottom, right, top margin
                            b: 60, r: 10, t: 20
                        },
                        hovermode: "closest"
                    
                        };

                        Plotly.newPlot(document.getElementById(row.ID), data, layout, {displaylogo: false});
                    
            
        });
};
