// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
window.onload = function() {

    /* data_tb */


    function unpack(rows, key) {
        return rows.map(function(row) { 
        return row[key]; 
        });
    }
    


    function generate_paracoords(data_tb, keys=['rheobase_thres', 'rheobase_width', 'rheobase_latency'], color='rheobase_thres') {
        //create out plotly fr
        var data = [{
            type: 'parcoords',
            line: {
            colorscale: 'Plotly',
            color: unpack(data_tb, color)
            },
        
            dimensions: keys.map(function (key) {
            return {
                //range: [Math.min(...unpack(data_tb, key)), Math.max(...unpack(data_tb, key))],
                label: key,
                values: unpack(data_tb, key)
            }
            })
        }]; // create the data object
        
        var layout = {
           
        };
        
        Plotly.newPlot('graphDiv_parallel', data, layout, {responsive: true}); // create the plot
        var graphDiv_parallel = document.getElementById("graphDiv_parallel") // get the plot div
        // graphDiv_parallel.on('plotly_restyle', function(data){
        //     var keys = []
        //     var ranges = []

        //     graphDiv_parallel.data[0].dimensions.forEach(function(d) {
        //             if (d.constraintrange === undefined){
        //                 keys.push(d.label);
        //                 ranges.push([-9999,9999]);
        //             }
        //             else{
        //                 keys.push(d.label);
        //                 var allLengths = d.constraintrange.flat();
        //                 if (allLengths.length > 2){
        //                     ranges.push([d.constraintrange[0][0],d.constraintrange[0][1]]); //return only the first filter applied per feature

        //                 }else{
        //                     ranges.push(d.constraintrange);
        //                 }
                        
                        
        //             } // => use this to find values are selected
        //     })

        //     filterByPlot(keys, ranges)
        // }); 
    };

    //encode labels
    function encode_labels(data, label) {
        var labels = data.map(function (a) { return a[label] });
        var unique_labels = [...new Set(labels)];
        var encoded_labels = labels.map(function (a) { return unique_labels.indexOf(a) });
        return [encoded_labels, unique_labels];
    };


    //umap plot
    function generate_umap(rows, keys=['Umap X', 'Umap Y', 'label']) {
        
        var colors = ['#f59582', '#f0320c', '#274f1b', '#d6b376', '#18c912', '#58d4d6', '#071aeb', '#000000']
        var encoded_labels = encode_labels(rows, keys[2]);
        // make a trace array for each label
        var traces = []
        encoded_labels[1].forEach(function (label, i) {traces.push(new Object())});

        // loop through the rows and append the data to the correct trace/data
        rows.forEach(function (row) {
            var trace = encoded_labels[1].indexOf(row[keys[2]]);
            if (traces[trace].x === undefined) {
                traces[trace].x = [];
                traces[trace].y = [];
                traces[trace].name = encoded_labels[1][trace];
                traces[trace].mode = 'markers';
                traces[trace].marker = { color: colors[trace], size: 5 };
            }
            traces[trace].x.push(row[keys[0]]);
            traces[trace].y.push(row[keys[1]]);
        });

        // create the data array
        var data = traces;

        var layout = {dragmode: 'lasso',autosize: true,
            margin: {                           // update the left, bottom, right, top margin
                b: 20, r: 10, t: 20
            },};

        Plotly.react('graphDiv_scatter', data, layout, { responsive: true });
        var graphDiv5 = document.getElementById("graphDiv_scatter")
        // graphDiv5.on('plotly_selected', function (eventData) {
        //     var ids = []
        //     var ranges = []
        //     if (typeof eventData !== 'undefined') {
        //         eventData.points.forEach(function (pt) { 
        //             ids.push(parseInt(pt.data.name));
        //         });
        //     }
        //     else {
        //         console.log(ids)
        //         ids = undefined
        //     }
        //     filterByID(ids);
        // });
    };
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


    /* onload */
    
    //find the elements of 
    var drop_parent = document.getElementById("umap-drop-menu");
    //get the children and add the event listener
    var drop_children = drop_parent.children;
    for (var i = 0; i < drop_children.length; i++) {
        drop_children[i].addEventListener('click', function (e) {
            var selected = e.target.innerHTML
            var keys = ['Umap X', 'Umap Y', selected]
            generate_umap(data_tb, keys);
        });
    }
    // create the table
    $table.bootstrapTable('load', data_tb)
    // while we are here, set the attr 'data-detail-formatter' to the function we defined above
    // refresh the table
    // set the table to be responsive
    //$table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
    $table.bootstrapTable('refresh')

    

};
