// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
$( document ).ready(function() {

    /* data_tb */

    /* colors */

    /* ekeys */

    /* para_keys */

    var restyle_programmatically = false;

    function unpack(rows, key) {
        return rows.map(function(row) { 
        return row[key]; 
        });
    }
    function filterByID(ids) {
        if (ids === undefined) {
            $('#table').bootstrapTable('filterBy', {})
            crossfilter(data_tb, [], "scatter");
        }
        else {
            $('#table').bootstrapTable('filterBy', { ID: ids })
            crossfilter(data_tb, ids, "scatter");
        }
    }

    function table_concatenator(labels){

        //update the global table data_tb with the selected labels
        data_tb = []
        labels.forEach(function(label) {
            // Assuming you have a way to get data for each label
            // For example, you might have a function getDataForLabel(label)
            var dataForLabel = subtables[label]
            // Concatenate or merge dataForLabel into data_tb
            data_tb.push(...dataForLabel);
        });
    }


    function generate_paracoords(data_tb, keys=['rheobase_thres', 'rheobase_width', 'rheobase_latency'], color='rheobase_thres', filter=[]) {
        //create out plotly fr

        if (filter.length > 0) {
            //get the row indices that match the filter
            var indices = data_tb.map(function (a) { return a['ID']; });
            indices = indices.filter(function (value, index) { return filter.includes(value); });
            var data_para = data_tb;
        }
        else {
            var indices = data_tb.map(function (a) { return a['ID']; });
            var data_para = data_tb;
        }

        color_vals = unpack(data_para, color)
        //encode the labels if they are strings
        if (typeof color_vals[0] === 'string') {
            var encoded_labels = encode_labels(data_para, color);
            color_vals = encoded_labels[0];
        }
        //check if the color key is in embed_colors
        if (Object.keys(embed_colors).includes(color)) {
            colorscale =  embed_colors[color];
            // affix a white color to the start of the colorscale
            
            //colorscale needs to be mapped to a range of 0-1 of the normalized values
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return ((el - min) / (max - min)); });
            //weird hack but multiply color_vals by 0.999 to avoid the last color in the colorscale
            //color_vals = color_vals.map(function (el) { return el * (1 - (1/colorscale.length)); });
            //colorscale.push('7f7f7f');
            //colorscale needs to be in the form [0, 'hex'], [1, 'hex'] ...
            colorscale = colorscale.map(function (el, i) { return [(i / (colorscale.length - 1)), "#"+el]; }); 
        }
        else {
            colorscale =  Plotly.d3.scale.category10();
        }
    
        //filter color_vals by our indices
        color_vals = color_vals.filter((el, i) => indices.includes(data_para[i]['ID']))
        .map(el => el);

        var data = [{
            type: 'parcoords',
            line: {
            colorscale: colorscale,
            color: color_vals,
            cauto: false,
            cmin: 0,
            cmax: 1,
            },
            ids: unpack(data_para, 'ID'),
        
            dimensions: keys.map(function (key) {
                values = unpack(data_para, key)
                //check if its a string
                if (typeof values[0] === 'string'){
                    //encode the labels
                    var encoded_labels = encode_labels(data_para, key);
                    if (encoded_labels[1].length < 2){
                        range = [-1, 1];
                    } else {
                        range = [0, encoded_labels[1].length - 1];
                    }

                    values = encoded_labels[0];
                    //filter by our indices
                    // Filter out elements not in indices and then map
                    values = values.filter((el, i) => indices.includes(data_para[i]['ID']))
                    .map(el => el);

                    var out = {
                        range: range,
                        tickvals: [...Array(encoded_labels[1].length).keys()],
                        ticktext: encoded_labels[1],
                        label: key,
                        values: values,
                        multiselect: true
                    }
                    
                } else {
                //replace null / nan with the mean
                mean_values = values.reduce((a, b) => a + b, 0) / values.length;
                //unfiltered_vals = unpack(data_tb, key);
                values = values.map(function (el) { return el == null || el != el ? mean_values : el; });
                //filter by our indices
                values = values.filter((el, i) => indices.includes(data_para[i]['ID']))
                .map(el => el);
                //unfiltered_vals = unfiltered_vals.map(function (el) { return el == null || el != el ? mean_values : el; });
                var out = {
                    range: [Math.min(...values), Math.max(...values)],
                    label: key,
                    values: values,
                    multiselect: false
                    }
                }
                return out
            }),
            labelangle: 0,
            labelside: 'bottom',
        }]; // create the data object
        
        var layout = {margin: {                           // update the left, bottom, right, top margin
            b: 90, r: 40, t: 90, l: 40
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

        var layout = {dragmode: 'lasso',
            autosize: true,
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
        crossfilter(data_tb, result, "parallel");
    };

    function crossfilter(data_tb, IDs, sender='') { 
        //set the restyle flag to true
        restyle_programmatically = true; //this way we can avoid the plotly_restyle event loop
        var graphDiv_parallel = document.getElementById("graphDiv_parallel");
        if (sender == "parallel") {
            //now we want to get the embedded graphDiv
            var graphDiv_scatter = document.getElementById("graphDiv_scatter");

            console.log("Crossfiltering data...");
            var selected = [];
            for (var i = 0; i < graphDiv_scatter.data.length; i++) {
                var trace = graphDiv_scatter.data[i];
                //figure out if trace.text is in the selected IDs
                var trace_selectedIndices = trace.text.map(function(value, index) {
                    return IDs.includes(value) ? index : undefined;
                }).filter(function(index) {
                    return index !== undefined;
                });
                //update the selected array
                selected.push(trace_selectedIndices);
            }
            //now we want to update the layout
            Plotly.update(graphDiv_scatter, {'selectedpoints': selected});
        } else if (sender == "scatter") {
            //in this case we completely reset the parallel plot
            generate_paracoords(data_tb, paracoordskeys, paracoordscolors, IDs);
        } else {
            //do nothing
        };
        //set the restyle flag to false
        restyle_programmatically = false

        

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
    //this is a bootsrap select
    
    //add an event listener
    drop_parent.addEventListener('change', function (e) {
        var selected = $('input[name="label-select"]:checked').val();
        var keys = ['Umap X', 'Umap Y', selected]
        generate_umap(data_tb, keys);
        //generate_paracoords(data_tb, keys, selected);
    });

    //listen for changes
    var dataset_parent = document.getElementById("dataset-select");
    // Check if the element has the class 'visually-hidden'
    if (dataset_parent.classList.contains('visually-hidden')) {
        console.log("Element is visually hidden");
    } else {
        var drop_parent = document.getElementById("dataset-drop-menu");
        //this is a bootsrap select
        
        //add an event listener
        drop_parent.addEventListener('change', function (e) {
            var selectedCheckboxes = document.querySelectorAll('input[name="dataset-select"]:checked');
            var selectedValues = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
            table_concatenator(selectedValues);
            //complete refresh
            var selected = $('input[name="label-select"]:checked').val();
            generate_umap(data_tb, ['Umap X', 'Umap Y', selected]);
            generate_paracoords(data_tb, para_keys, paracoordscolors)
            $table.bootstrapTable('load', data_tb)
            //$table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
            $table.bootstrapTable('refresh')
        });

    }




    //add an event listener for table changes
    $table.on('all.bs.table', function (e, name, args) {
        generate_plots();
    });

    generate_plots();

    // refresh the table
    // set the table to be responsive


    //now create our cell plots
    

});
