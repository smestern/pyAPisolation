// Dynamic mode template script for pyAPisolation webViz
// This script extends template_common.js to add AJAX-based trace loading
// ⚠️ EXPERIMENTAL - Requires running Flask server

// Include common functions (loaded via script tag in HTML)
// <script src="./assets/template_common.js"></script>

$( document ).ready(function() {

    /* data_tb */

    /* colors */

    /* ekeys */

    /* para_keys */

    /* umap_labels */

    /* dataset_label_col */

    /* table_links */

    var table_concat = false;
    var restyle_programmatically = false;
    var pre_selected_datasets = [];
    var prev_ranges = {};
    var prev_filter = "";
    var prev_parallel_IDs = null;

    // ============ Dynamic Mode Plot Functions (AJAX Loading) ============

    function maketrace(row){
        var foldername = row['foldername.1'] || row['foldername'] || '';
        var url = "/api/" + row.ID + "?foldername=" + encodeURIComponent(foldername);
        
        $.ajax({
            url: url,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                // data is a 2D array: first row is time, subsequent rows are traces
                plotTraceData(row.ID, data, 'graphDiv_'+row.ID+'_plot');
            },
            error: function(xhr, status, error) {
                console.error('Error loading trace for ' + row.ID + ':', error);
                var div = document.getElementById("graphDiv_"+row.ID+"_plot");
                if (div) {
                    div.innerHTML = '<p class="text-danger">Error loading trace</p>';
                }
            }
        });
    }

    function plotTraceData(id, data, divId) {
        // data[0] is time, data[1:] are voltage traces
        if (!data || data.length < 2) {
            console.error('Invalid trace data for ' + id);
            return;
        }

        var time = data[0];
        var traces = [];
        
        // Create a trace for each sweep
        for (var i = 1; i < data.length; i++) {
            traces.push({
                x: time,
                y: data[i],
                type: 'scattergl',  // WebGL for better performance
                mode: 'lines',
                name: 'Sweep ' + i,
                line: {width: 1},
                showlegend: false
            });
        }

        var layout = {
            autosize: true,
            height: 200,
            margin: {l: 40, r: 20, t: 20, b: 30},
            xaxis: {title: 'Time (s)'},
            yaxis: {title: 'mV'},
            hovermode: 'closest'
        };

        Plotly.newPlot(divId, traces, layout, {responsive: true, displayModeBar: false});
    }

    function makefi(row){
        // F-I curve can also be loaded dynamically if available
        var foldername = row['foldername.1'] || row['foldername'] || '';
        var url = "/api/" + row.ID + "_FI?foldername=" + encodeURIComponent(foldername);
        
        $.ajax({
            url: url,
            method: 'GET',
            dataType: 'json',
            success: function(data) {
                plotFIData(row.ID, data, 'graphDiv_'+row.ID+'_plot_fi');
            },
            error: function(xhr, status, error) {
                // F-I curve might not exist for all cells, fail silently
                console.log('F-I curve not available for ' + row.ID);
                var div = document.getElementById("graphDiv_"+row.ID+"_plot_fi");
                if (div) {
                    div.innerHTML = '<p class="text-muted">F-I not available</p>';
                }
            }
        });
    }

    function plotFIData(id, data, divId) {
        // data format: {current: [...], rate: [...]}
        if (!data || !data.current || !data.rate) {
            console.error('Invalid F-I data for ' + id);
            return;
        }

        var trace = {
            x: data.current,
            y: data.rate,
            type: 'scattergl',  // WebGL for better performance
            mode: 'lines+markers',
            marker: {size: 8, color: 'black'},
            line: {width: 2, color: 'black'}
        };

        var layout = {
            autosize: true,
            height: 200,
            margin: {l: 40, r: 20, t: 20, b: 30},
            xaxis: {title: 'Current (pA)'},
            yaxis: {title: 'Firing Rate (Hz)'}
        };

        Plotly.newPlot(divId, [trace], layout, {responsive: true, displayModeBar: false});
    }

    // ============ Plotly Visualization Functions (from template.js) ============

    function generate_paracoords(data_tb, keys=['rheobase_thres', 'rheobase_width', 'rheobase_latency'], color='rheobase_thres', filter=[]) {
        if (filter.length > 0) {
            var indices = data_tb.map(function (a) { return a['ID']; });
            indices = indices.filter(function (value, index) { return filter.includes(value); });
            var data_para = data_tb;
        }
        else {
            var indices = data_tb.map(function (a) { return a['ID']; });
            var data_para = data_tb;
        }

        color_vals = unpack(data_para, color)
        var colorscale = 'Portland';
        
        if (typeof color_vals[0] === 'string') {
            var encoded_labels = encode_labels(data_para, color);
            color_vals = encoded_labels[0];
        }
        
        if (Object.keys(embed_colors).includes(color)) {
            colorscale = embed_colors[color];
            const encodedKeys = encoded_labels[1];
            colorscale = Object.fromEntries(
                Object.entries(colorscale).filter(([key, value]) => encodedKeys.includes(key))
            );
            
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return ((el - min) / (max - min)); });
            colorscale = Object.values(colorscale)
            colorscale = colorscale.map(function (el, i) { return [(i / (colorscale.length - 1)), "#"+el]; });
        }
        else {
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return ((el - min) / (max - min)); });
        }
    
        color_vals = color_vals.filter((el, i) => indices.includes(data_para[i]['ID']));

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
            customdata: unpack(data_para, 'ID'),
        
            dimensions: keys.map(function (key) {
                values = unpack(data_para, key)
                if (typeof values[0] === 'string'){
                    var encoded_labels = encode_labels(data_para, key);
                    if (encoded_labels[1].length < 2){
                        range = [-1, 1];
                    } else {
                        range = [0, encoded_labels[1].length - 1];
                    }
                    values = encoded_labels[0];
                    values = values.filter((el, i) => indices.includes(data_para[i]['ID']));

                    var out = {
                        range: range,
                        tickvals: [...Array(encoded_labels[1].length).keys()],
                        ticktext: encoded_labels[1],
                        label: key,
                        values: values,
                        multiselect: true
                    }
                } else {
                    mean_values = values.reduce((a, b) => a + b, 0) / values.length;
                    values = values.map(function (el) { return el == null || el != el ? mean_values : el; });
                    values = values.filter((el, i) => indices.includes(data_para[i]['ID']));
                    var out = {
                        range: [Math.min(...values), Math.max(...values)],
                        label: key,
                        values: values,
                        multiselect: false
                    }
                }
                return out
            }),
            labelangle: 45,
            labelside: 'bottom',
        }];
        
        var layout = {
            autosize: true,
            height: 300,
            margin: {b: 120, r: 40, t: 30, l: 40},
        };
        
        fig = Plotly.newPlot('graphDiv_parallel', data, layout, {responsive: true, displayModeBar: false});
        var graphDiv_parallel = document.getElementById("graphDiv_parallel")
        graphDiv_parallel.on('plotly_restyle', function(data){
            // Prevent event loop when restyle is triggered programmatically
            if (restyle_programmatically) {
                return;
            }
            
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
                    if (d.ticktext !== undefined){
                        var selected = d.tickvals.filter(function(value, index) { return (d.constraintrange[0] <= value && d.constraintrange[1] >= value); });
                        var selected_text = selected.map(function(value, index) { return d.ticktext[value]; });
                        ranges.push(selected_text);
                    } else {
                        if (allLengths.length > 2){
                            ranges.push([d.constraintrange[0][0],d.constraintrange[0][1]]);
                        } else {
                            ranges.push(d.constraintrange);
                        }
                    }   
                }
            })
            filterByPlot(keys, ranges);
        }); 
    }

    function generate_umap(rows, keys=['Umap X', 'Umap Y', 'label'], colors=embed_colors, dataset_en='Species', dataset_shapes=['circle', 'x', 'square', 'triangle-up', 'triangle-down', 'diamond', 'cross'], dataset_opacity=[0.25, 1]) {
        var encoded_labels = encode_labels(rows, keys[2]);
        var encoded_dataset = encode_labels(rows, dataset_en);
        
        if (Object.keys(colors).includes(keys[2])) {
            label_color = colors[keys[2]];
        } else {
            label_color = Plotly.d3.scale.category10().range();
        }
        
        var traces = [];
        
        if (isContinuousFloat(encoded_labels[1])) {
            traces.push({
                x: [],
                y: [],
                text: [],
                customdata: [],
                mode: 'markers',
                name: 'Continuous Data',
                marker: { color: unpack(rows, keys[2]), size: 5, symbol: 'circle', 
                    colorscale: 'Portland', showscale: true, 
                    colorbar: { title: {text: keys[2]}} }
            });
        } else {
            encoded_labels[1].forEach(function (label, i) {
                if (keys[2] != dataset_en && encoded_dataset[1].length > 1) {
                    encoded_dataset[1].forEach(function (dataset, j) {
                        traces.push({
                            x: [], y: [], text: [], customdata: [],
                            mode: 'markers',
                            name: `${label} - ${dataset}`,
                            marker: { color: label_color[label], size: 5, symbol: dataset_shapes[j], opacity: dataset_opacity[j] }
                        });
                    });
                } else {
                    traces.push({
                        x: [], y: [], text: [], customdata: [],
                        mode: 'markers',
                        name: `${label}`,
                        marker: { color: label_color[label], size: 5, symbol: 'circle' }
                    });
                }
            });
        }

        rows.forEach(function (row) {
            if (isContinuousFloat(encoded_labels[1])) {
                traces[0].x.push(row[keys[0]]);
                traces[0].y.push(row[keys[1]]);
                traces[0].text.push(row['ID']);
                traces[0].customdata.push(row['ID']);
            } else {
                if (keys[2] != dataset_en && encoded_dataset[1].length > 1) {
                    var traceIndex = encoded_labels[1].indexOf(row[keys[2]]) * encoded_dataset[1].length + encoded_dataset[1].indexOf(row[dataset_en]);
                } else {
                    var traceIndex = encoded_labels[1].indexOf(row[keys[2]]);
                }
                if (encoded_labels[1][traceIndex] != 'nan') {
                    traces[traceIndex].x.push(row[keys[0]]);
                    traces[traceIndex].y.push(row[keys[1]]);
                    traces[traceIndex].text.push(row['ID']);
                    traces[traceIndex].customdata.push(row['ID']);
                }
            }
        });

        var layout = {
            dragmode: 'lasso',
            autosize: true,
            margin: {b: 20, r: 20, t: 20, l: 20},
            xaxis: { zeroline: false },
            yaxis: { zeroline: false },
            legend: {x: 1, y: 0.5},
            scene: {aspectmode: "cube", xaxis: {title: keys[0]}, yaxis: {title: keys[1]}}
        };

        if (traces.length == 1) {
            layout.showlegend = false;
        }

        Plotly.react('graphDiv_scatter', traces, layout, { responsive: true });
        var graphDiv5 = document.getElementById("graphDiv_scatter")
        graphDiv5.on('plotly_selected', function (eventData) {
            var ids = []
            if (typeof eventData !== 'undefined') {
                eventData.points.forEach(function (pt) { 
                    ids.push(pt.text);
                });
            }
            else {
                ids = undefined
            }
            filterByID(ids);
        });
    }

    // ============ Plot Generation and Table Management ============

    function generate_plots() {
        console.log("Generating plots (dynamic mode)...");
        $table.bootstrapTable('showLoading');
        var rows = $table.bootstrapTable('getData', {useCurrentPage: true});
        let promises = rows.map(row => {
            return new Promise(resolve => {
                setTimeout(() => {
                    maketrace(row);
                    makeephys(row);
                    makefi(row);
                    makeLink(row);
                    resolve();
                }, 100); // Shorter timeout for AJAX loading
            });
        });
    
        Promise.all(promises).then(() => {
            $table.bootstrapTable('hideLoading');
        });
    }

    // ============ Table Initialization ============

    var $table = $('#table')
    $table.bootstrapTable('load', data_tb)
    $table.bootstrapTable('refreshOptions', {
        pagination: true,
        pageSize: 50,
        pageList: [25, 50, 100, 200]
    })
    $table.bootstrapTable('refresh')

    /* onload */
    
    var drop_parent = document.getElementById("umap-drop-menu");
    drop_parent.addEventListener('change', function (e) {
        var selected = $('input[name="label-select"]:checked').val();
        if ($('input[name="label-select"]:checked')[0].classList.contains('multi-key')) {
            selected = $('input[name="label-select"]:checked')[0].nextElementSibling.children[0].value;
        }

        if (selected === split_strs) {
            var selectedCheckboxes = document.querySelectorAll('input[name="dataset-select"]:checked');
            var selectedValues = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
            pre_selected_datasets = selectedValues;
            var checkboxes = document.querySelectorAll('input[name="dataset-select"]');
            checkboxes.forEach(checkbox => checkbox.checked = true);
            dataset_selector();
        } else if (pre_selected_datasets.length > 0){ 
            pre_selected_datasets.forEach(function(dataset){
                var checkbox = document.getElementById(dataset);
                checkbox.checked = true;
            });
            var checkboxes = document.querySelectorAll('input[name="dataset-select"]');
            checkboxes.forEach(checkbox => {
                if (!pre_selected_datasets.includes(checkbox.value)){
                    checkbox.checked = false;
                }
            });
            pre_selected_datasets = [];
            dataset_selector();
        } else {
            var keys = ['Umap X', 'Umap Y', selected]
            generate_umap(data_tb, keys);
            generate_paracoords(data_tb, paracoordskeys, selected) 
        }
    });

    var dataset_parent = document.getElementById("dataset-select");
    if (!dataset_parent.classList.contains('visually-hidden')) {
        var drop_parent = document.getElementById("dataset-drop-menu");
        drop_parent.addEventListener('change', function (e) {
            dataset_selector();
        });
    }

    $table.on('all.bs.table', function (e, name, args) {
        if (name == "click-cell.bs.table" || name == "click-row.bs.table" || name == "dbl-click-row.bs.table"){ 
            return;
        } else {
            generate_plots();
        }
    });

    generate_plots();
});
