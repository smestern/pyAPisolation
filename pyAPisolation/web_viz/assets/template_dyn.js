// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
window.onload = function() {
    function unpack(rows, key) {
        return rows.map(function(row) { 
        return row[key]; 
        });
    }

    //create out plotly fr
    var data = [{
        type: 'parcoords',
        pad: [80,80,80,80],
        line: {
        colorscale: 'Plotly',
        color: unpack(data_tb, 'label')
        },
    
        dimensions: [{
        label: 'rheobase_thres',
        values: unpack(data_tb, 'rheobase_thres')
        }, {
        label: 'rheobase_width',
        values: unpack(data_tb, 'rheobase_width')
        },{
        label: 'rheobase_latency',
        values: unpack(data_tb, 'rheobase_latency')
        },{
        label: 'label',
        values: unpack(data_tb, 'label')
        }
        
        
        ]
    }]; // create the data object
    
    var layout = {
        width: 1200
    };
    
    Plotly.newPlot('graphDiv', data, layout, {displaylogo: false}, {responsive: true}); // create the plot
    var graphDiv = document.getElementById("graphDiv") // get the plot div
    graphDiv.on('plotly_restyle', function(data){
        var keys = []
        var ranges = []

        graphDiv.data[0].dimensions.forEach(function(d) {
                if (d.constraintrange === undefined){
                    keys.push(d.label);
                    ranges.push([-9999,9999]);
                }
                else{
                    keys.push(d.label);
                    var allLengths = d.constraintrange.flat();
                    if (allLengths.length > 2){
                        ranges.push([d.constraintrange[0][0],d.constraintrange[0][1]]); //return only the first filter applied per feature

                    }else{
                        ranges.push(d.constraintrange);
                    }
                    
                    
                } // => use this to find values are selected
        })

        filterByPlot(keys, ranges)
    }); 


    //table functions
    function traceFormatter(index, row) {
        var html = []
        
        
        
        html.push('<div id="' + row.ID + '"></div>');
        
        html.push('</div>');
        
        setTimeout(() =>{maketrace(row)}, 1000);
        return html.join('');
    }
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
                        width: 950,
                        height: 500,
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
    function filterByPlot(keys, ranges){		
    var ids = []
    var fildata = data
    var newArray = data_tb.filter(function (el) {
            return el.rheobase_thres <= ranges[0][1] &&
        el.rheobase_thres >= ranges[0][0] &&
        el.rheobase_width <= ranges[1][1] &&
        el.rheobase_width >= ranges[1][0] &&
        el.rheobase_latency <= ranges[2][1] &&
        el.rheobase_latency >= ranges[2][0] &&
        el.label <= ranges[3][1] &&
        el.label >= ranges[3][0];	

        });
    let result = newArray.map(function(a) { return a.ID; });

    $('#table').bootstrapTable('filterBy',{'ID': result})
    };
    function cellStyle(value, row, index) {
    var classes = [
    'bg-blue',
    'bg-green',
    'bg-orange',
    'bg-yellow',
    'bg-red'
    ]

    if (value > 0) {
        return {
            css: {
                'background-color': 'hsla(0, 100%, 50%,' + (value/40) + ')'
            }
        }
    }
    return {
        css: {
            color: 'black'
        }
    }
    }




    // create the table
    var data = data_tb
    // find the table div
    var $table = $('#table')

    // create the table
    $table.bootstrapTable('load', data)
    // while we are here, set the attr 'data-detail-formatter' to the function we defined above
    // refresh the table
    // set the table to be responsive
    $table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
    $table.bootstrapTable('refresh')
    

};
