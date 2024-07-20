// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
$( document ).ready(function() {

    /* data_tb */

    /* colors */

    /* ekeys */


    var restyle_programmatically = false;

    function unpack(rows, key) {
        return rows.map(function(row) { 
        return row[key]; 
        });
    }
    function filterByID(ids) {
        if (ids === undefined) {
            $('#table').bootstrapTable('filterBy', {})
        }
        else {
            $('#table').bootstrapTable('filterBy', { ID: ids })
        }
    }


    function generate_paracoords(data_tb, keys=['rheobase_thres', 'rheobase_width', 'rheobase_latency'], color='rheobase_thres') {
        //create out plotly fr

        color_vals = unpack(data_tb, color)
        //encode the labels if they are strings
        if (typeof color_vals[0] === 'string') {
            var encoded_labels = encode_labels(data_tb, color);
            color_vals = encoded_labels[0];
        }
        //check if the color key is in embed_colors
        if (Object.keys(embed_colors).includes(color)) {
            colorscale =  embed_colors[color];
            //colorscale needs to be mapped to a range of 0-1 of the normalized values
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return (el - min) / (max - min); });
            //colorscale needs to be in the form [0, 'hex'], [1, 'hex'] ...
            colorscale = colorscale.map(function (el, i) { return [i / (colorscale.length - 1), "#"+el]; });
        }
        else {
            colorscale =  Plotly.d3.scale.category10();
        }
    

        var data = [{
            type: 'parcoords',
            line: {
            colorscale: colorscale,
            color: color_vals
            },
        
            dimensions: keys.map(function (key) {
                values = unpack(data_tb, key)
                //check if its a string
                if (typeof values[0] === 'string'){
                    //encode the labels
                    var encoded_labels = encode_labels(data_tb, key);
                    var out = {
                        range: [0, encoded_labels[1].length - 1],
                        tickvals: [...Array(encoded_labels[1].length).keys()],
                        ticktext: encoded_labels[1],
                        label: key,
                        values: encoded_labels[0],
                        multiselect: true
                    }
                    
                } else {
                //replace null / nan with the mean
                mean_values = values.reduce((a, b) => a + b, 0) / values.length;
                values = values.map(function (el) { return el == null || el != el ? mean_values : el; });
                var out = {
                    range: [Math.min(...values), Math.max(...values)],
                    label: key,
                    values: values,
                    multiselect: false
                    }
                }
                return out
            }),
            labelangle: -45
        }]; // create the data object
        
        var layout = {margin: {                           // update the left, bottom, right, top margin
            b: 20, r: 40, t: 90, l: 20
        },
        };
        
        fig = Plotly.newPlot('graphDiv_parallel', data, layout, {responsive: true, displayModeBar: false}); // create the plots
        var graphDiv_parallel = document.getElementById("graphDiv_parallel") // get the plot div
        graphDiv_parallel.on('plotly_restyle', function(data){
            var keys = []
            var ranges = []

            graphDiv_parallel.data[0].dimensions.forEach(function(d) {
                    if (d.constraintrange === undefined){
                        keys.push(d.label);
                        ranges.push([-9999,9999]);
                    }
                    else{
                        keys.push(d.label);
                        var allLengths = d.constraintrange.flat();
                        //check if the label is actually categorical, by looking at ticktext
                        if (d.ticktext !== undefined){
                            //find the tickvals that are selected
                            var selected = d.tickvals.filter(function(value, index) { return (d.constraintrange[0] <= value && d.constraintrange[1] >= value); });
                            //find the ticktext that corresponds to the tickvals
                            var selected_text = selected.map(function(value, index) { return d.ticktext[value]; });
                            ranges.push(selected_text);
                            
                        }else {
                            if (allLengths.length > 2){
                            ranges.push([d.constraintrange[0][0],d.constraintrange[0][1]]); //return only the first filter applied per feature

                            }else{
                                ranges.push(d.constraintrange);
                            }
                        }   
                    } // => use this to find values are selected
            })

            filterByPlot(keys, ranges);
        }); 
    };

    //encode labels
    function encode_labels(data, label) {
        var labels = data.map(function (a) { return a[label] });
        var unique_labels = [...new Set(labels)];
        var encoded_labels = labels.map(function (a) { return unique_labels.indexOf(a) });
        return [encoded_labels, unique_labels];
    };


    //umap plot
    function generate_umap(rows, keys=['Umap X', 'Umap Y', 'label'], colors=embed_colors) {
        
        var encoded_labels = encode_labels(rows, keys[2]);
        if (Object.keys(colors).includes(keys[2])) {
            label_color =  colors[keys[2]];
        }
        else {
            label_color =  Plotly.d3.scale.category10().range();
        }
        // make a trace array for each label
        var traces = []
        encoded_labels[1].forEach(function (label, i) {traces.push(new Object())});

        // loop through the rows and append the data to the correct trace/data
        rows.forEach(function (row) {
            var trace = encoded_labels[1].indexOf(row[keys[2]]);
            if (encoded_labels[1][trace] != 'nan'){
                if (traces[trace].x === undefined) {
                    
                        traces[trace].x = [];
                        traces[trace].y = [];
                        traces[trace].text = [];
                        traces[trace].name = encoded_labels[1][trace];
                        traces[trace].mode = 'markers';
                        traces[trace].marker = { color: label_color[trace], size: 5 };
                    
                }
                traces[trace].x.push(row[keys[0]]);
                traces[trace].y.push(row[keys[1]]);
                traces[trace].text.push(row['ID']);
            };
        });

        // create the data array
        var data = traces;

        var layout = {dragmode: 'lasso',autosize: true,
            margin: {                           // update the left, bottom, right, top margin
                b: 20, r: 20, t: 20, l: 20
            },
            legend: {
                x: 1,
                xanchor: 'right',
                yanchor: 'top',
                y: 0.2
            },
            scene: {aspectmode: "cube", xaxis: {title: keys[0]}, yaxis: {title: keys[1]}}
        };

        Plotly.react('graphDiv_scatter', data, layout, { responsive: true, });
        var graphDiv5 = document.getElementById("graphDiv_scatter")
        graphDiv5.on('plotly_selected', function (eventData) {
            var ids = []
            var ranges = []
            if (typeof eventData !== 'undefined') {
                eventData.points.forEach(function (pt) { 
                    ids.push(pt.text);
                });
            }
            else {
                console.log(ids)
                ids = undefined
            }
            filterByID(ids);
        });
    };-

    //table functions
    function traceFormatter(index, row) {
        var html = []
        
        
        
        html.push('<div id="' + row.ID + '"></div>');
        
        html.push('</div>');
        
        setTimeout(() =>{maketrace(row)}, 1000);
        return html.join('');
    }

    function plotFormatter(index, row) {

        return '<div id="' + row.ID + '"></div>';
    };


    function maketrace(row){
        var url = "./data/traces/" + row.ID + ".svg"
        var html = []
        html.push('<img src="' + url + '" alt="Traces">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot")
        div.innerHTML = html.join('');
        
    };
    function makerheo(row){
        var url = "./data/traces/" + row.ID + "_rheo.png"
        var html = []
        html.push('<img src="' + url + '" alt="Rheobase" style="width: 10%">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot_rheo")
        div.innerHTML = html.join('');
    };
    function makefi(row){
        var url = "./data/traces/" + row.ID + "_FI.svg"
        var html = []
        html.push('<img src="' + url + '" alt="FI">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot_fi")
        div.innerHTML = html.join('');
    };

    function makeephys(row, keys=ekeys){
        var html = [];
        
        //we just need to populate it with rows
        html.push('<div class="col table-ephys">');
        
        //loop through the keys
        keys.forEach(function(key){
            html.push('<div class="row">');
            html.push('<span class="ephys-key">'+key+'</span>');
            html.push('<span class="ephys-value"> '+row[key]+'</span>');
            html.push('</div>');
        
            
        });
        html.push('</div>');
        //get the div



        var div = document.getElementById("table_"+row.ID)
        div.innerHTML = html.join('');


    };


    function filterByPlot(keys, ranges){		
        var newArray = data_tb.filter(function (el) {
                return keys.every(function (key, i) {
                    if (ranges[i][0] == -9999){
                        return true;
                    }
                    else if (typeof ranges[i][0] === 'string'){
                        return ranges[i].includes(el[key]);
                    }
                    else{
                        return el[key] >= ranges[i][0] && el[key] <= ranges[i][1];
                    }
                });	
            });
        let result = newArray.map(function(a) { return a.ID; });

        $('#table').bootstrapTable('filterBy',{'ID': result});
        crossfilter(data_tb, result);
    };

    function crossfilter(data_tb, IDs) { 
        //set the restyle flag to true
        restyle_programmatically = true; //this way we can avoid the plotly_restyle event loop

        //now we want to get the embedded graphDiv
        var graphDiv_scatter = document.getElementById("graphDiv_scatter");

        console.log("Crossfiltering data...");
        var selected = [];
        for (var i = 0; i < graphDiv_scatter.data.length; i++) {
            var trace = graphDiv_scatter.data[i];
            //figure out if trace.text is in the selected IDs
            


    }


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

    function generate_plots() {
        console.log("Generating plots...");
        $table.bootstrapTable('showLoading');
        var rows = $table.bootstrapTable('getData', {useCurrentPage: true}); // get the rows, only the visible ones
        let promises = rows.map(row => {
            return new Promise(resolve => {
                setTimeout(() => {
                    maketrace(row);
                    // Uncomment the next line if makerheo should also be awaited
                    // makerheo(row);
                    makeephys(row);
                    makefi(row);
                    resolve();
                }, 1000);
            });
        });
    
        Promise.all(promises).then(() => {
            $table.bootstrapTable('hideLoading');
        });
    }

    // create the table
    // find the table div
    var $table = $('#table')
    // create the table
    //%table.bootstrapTable({data: data_tb})
    $table.bootstrapTable('load', data_tb)
    //$table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
    $table.bootstrapTable('refresh')

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
            //update the dropdown
            document.getElementById("drop-button").innerHTML = selected;
        });
    }

    // while we are here, set the attr 'data-detail-formatter' to the function we defined above

    //add an event listener for table changes
    $table.on('all.bs.table', function (e, name, args) {
        generate_plots();
    });

    generate_plots();

    // refresh the table
    // set the table to be responsive


    //now create our cell plots
    

});
