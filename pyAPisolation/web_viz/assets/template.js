
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
    
    //var layout = {
    //    width: 1200
    //};
    
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

    //umap plot
    function generate_umap(rows) {
        data = []
        m_trace_x = []
        m_trace_y = []
        h_trace_x = []
        h_trace_y = []
        var colors = ['#f59582', '#f0320c', '#274f1b', '#d6b376', '#18c912', '#58d4d6', '#071aeb', '#000000']
        rows.forEach(function (row) {
            var rowdata = Object.keys(row).map(function (e) {
                return row[e]
            })
            var timeseries = Object.keys(row);

            if ((row.labels.includes("Macaca") != true) && (row.labels.includes("Callithrix") != true)) {
                m_trace_y = m_trace_y.concat([row.X2])
                m_trace_x = m_trace_x.concat([row.X1])


            } else if (row.labels.includes("Macaca") || row.labels.includes("Callithrix")) {
                var sweepname = row.IDs
                var trace = {
                    type: 'scatter',                    // set the chart type
                    mode: 'markers',                      // connect points with lines
                    name: sweepname,
                    y: [row.X2],
                    x: [row.X1],

                };
                data.push(trace);
            }



        });
        var m_trace = {
            type: 'scatter',                    // set the chart type
            mode: 'markers',                      // connect points with lines
            name: 'mouse data',
            y: m_trace_y,
            x: m_trace_x,
            marker: {
                color: '#D3D3D3',
                opacity: 0.55,
            },

        };
        var h_trace = {
            type: 'scatter',                    // set the chart type
            mode: 'markers',                      // connect points with lines
            name: 'human data',
            y: h_trace_y,
            x: h_trace_x,
            marker: {
                color: '#707070',
                opacity: 0.55,
            }

        };
        data = data.concat(m_trace, h_trace);

        var layout = {
            dragmode: 'lasso',
            autosize: true,
            margin: {                           // update the left, bottom, right, top margin
                b: 20, r: 10, t: 20
            },

        };

        Plotly.react('graphDiv_scatter4', data, layout, { displaylogo: false }, { responsive: true });
        var graphDiv5 = document.getElementById("graphDiv_scatter4")
        graphDiv5.on('plotly_selected', function (eventData) {
            var ids = []
            var ranges = []
            if (typeof eventData !== 'undefined') {

                eventData.points.forEach(function (pt) {


                    ids.push(parseInt(pt.data.name));
                });
            }
            else {
                console.log(ids)
                ids = undefined
            }
            filterByID(ids);
        });
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
        var url = "./data/" + row.ID + ".csv"
        Plotly.d3.csv(url, function(rows){			
                        data = []
                        var i = 1
                        rows.forEach(function(row) {
                            
                            var sweepname = 'Sweep ' + i + ': ' + Math.round(row[Object.keys(row)[0]].toString()) + ' pA'
                            delete row[Object.keys(row)[0]]
                            var rowdata =  Object.keys(row).map(function(e) { 
                                                                    return row[e]
                                                                })
                            var timeseries = Object.keys(row);
                            
                            
                            var trace = {
                                    type: 'scattergl',                    // set the chart type
                                    mode: 'lines',                      // connect points with lines
                                    name: sweepname,
                                    y: rowdata,
                                    x: timeseries,
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
                            dtick: 0.25,
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
