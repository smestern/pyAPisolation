// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
$(document).ready(function () {

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

    var prev_rows = [];

    function unpack(rows, key) {
        return rows.map(function (row) {
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


    function isContinuousFloat(labels) {
        return labels.every(label => typeof label === 'number' || label === undefined || label === null);
    }

    function table_concatenator(labels) {

        //update the global table data_tb with the selected labels
        data_tb = []
        labels.forEach(function (label) {
            // Assuming you have a way to get data for each label
            var dataForLabel = subtables[label]
            //we also want rename the columns in the subcolumn with the split label
            // if (labels.length > 1) {
            //     table_concat = true;
            //     dataForLabel.forEach( function(row){
            //         umap_labels.forEach(function(ulabel){
            //             //check if labels are present
            //             if (row[ulabel].indexOf(label) < 0){
            //             row[ulabel] = row[ulabel] + " " + label};
            //         });
            //     });
            // } else {
            //     table_concat = false;
            //     dataForLabel.forEach( function(row){
            //         umap_labels.forEach(function(ulabel){
            //             //check if labels are present
            //             if (row[ulabel].indexOf(label) > 0){
            //             row[ulabel] = row[ulabel].substring(0, row[ulabel].indexOf((" " + label)))};
            //         });
            //     });

            // }
            // Concatenate or merge dataForLabel into data_tb
            data_tb.push(...dataForLabel);
        });
    }


    function generate_paracoords(data_tb, keys = ['rheobase_thres', 'rheobase_width', 'rheobase_latency'], color = 'rheobase_thres', filter = []) {
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
            colorscale = embed_colors[color];
            // affix a white color to the start of the colorscale
            //filter the colorscale to only colors found in encoded
            // Filter the colorscale to only include colors found in encoded_labels[1]
            const encodedKeys = encoded_labels[1];
            colorscale = Object.fromEntries(
                Object.entries(colorscale).filter(([key, value]) => encodedKeys.includes(key))
            );

            //colorscale needs to be mapped to a range of 0-1 of the normalized values
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return ((el - min) / (max - min)); });
            // colorscale is an object with keys being the actual label, and value being the color
            colorscale = Object.values(colorscale)
            colorscale = colorscale.map(function (el, i) { return [(i / (colorscale.length - 1)), "#" + el]; });;

        }
        else {
            // Generate a colorscale using Plotly's category10 scale
            var colorscale = 'Portland';

            // Normalize color values to a range of 0-1
            var min = Math.min(...color_vals);
            var max = Math.max(...color_vals);
            color_vals = color_vals.map(function (el) { return ((el - min) / (max - min)); });

            // Map the normalized values to the colorscale
            var colorMap = {};
            color_vals.forEach(function (val, index) {
                var colorIndex = Math.floor(val * (colorscale.length - 1));
                colorMap[data_para[index]['ID']] = colorscale[colorIndex];
            });
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
            customdata: unpack(data_para, 'ID'),

            dimensions: keys.map(function (key) {
                values = unpack(data_para, key)
                //check if its a string
                if (typeof values[0] === 'string') {
                    //encode the labels
                    var encoded_labels = encode_labels(data_para, key);
                    if (encoded_labels[1].length < 2) {
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
            labelangle: 45,
            labelside: 'bottom',
        }]; // create the data object

        var layout = {
            autosize: true,
            height: 300,
            margin: {                           // update the left, bottom, right, top margin
                b: 120, r: 40, t: 30, l: 40
            },
        };

        fig = Plotly.newPlot('graphDiv_parallel', data, layout, { responsive: true, displayModeBar: false }); // create the plots
        var graphDiv_parallel = document.getElementById("graphDiv_parallel") // get the plot div
        graphDiv_parallel.on('plotly_restyle', function (data) {
            // Prevent event loop when restyle is triggered programmatically
            if (restyle_programmatically) {
                return;
            }
            console.log("Parallel plot restyled, updating table...");
            var keys = []
            var ranges = []

            graphDiv_parallel.data[0].dimensions.forEach(function (d) {
                if (d.constraintrange === undefined) {
                    keys.push(d.label);
                    ranges.push([-9999, 9999]);
                }
                else {
                    keys.push(d.label);
                    var allLengths = d.constraintrange.flat();
                    //check if the label is actually categorical, by looking at ticktext
                    if (d.ticktext !== undefined) {
                        //find the tickvals that are selected
                        var selected = d.tickvals.filter(function (value, index) { return (d.constraintrange[0] <= value && d.constraintrange[1] >= value); });
                        //find the ticktext that corresponds to the tickvals
                        var selected_text = selected.map(function (value, index) { return d.ticktext[value]; });
                        ranges.push(selected_text);

                    } else {
                        if (allLengths.length > 2) {
                            ranges.push([d.constraintrange[0][0], d.constraintrange[0][1]]); //return only the first filter applied per feature

                        } else {
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
        var labelToIndex = {};
        var unique_labels = [];
        var encoded_labels = [];
        
        for (var i = 0; i < data.length; i++) {
            var value = data[i][label];
            if (!(value in labelToIndex)) {
                labelToIndex[value] = unique_labels.length;
                unique_labels.push(value);
            }
            encoded_labels.push(labelToIndex[value]);
        }
        
        return [encoded_labels, unique_labels];
    };


    //umap plot
    function generate_umap(rows, keys = ['Umap X', 'Umap Y', 'label'], colors = embed_colors, dataset_en = 'Species', dataset_shapes = ['o', 'x', 'square', 'triangle-up', 'triangle-down', 'diamond', 'cross', 'x', 'square', 'triangle-up', 'triangle-down', 'diamond', 'cross'], dataset_opacity = [0.25, 1]) {
        console.log("Generating UMAP plot...");

        var encoded_labels = encode_labels(rows, keys[2]);
        var encoded_dataset = encode_labels(rows, dataset_en);


        
        if (Object.keys(colors).includes(keys[2])) {
            label_color = colors[keys[2]];
        } else {
            // Use a fallback color palette compatible with newer Plotly versions
            label_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];
        }

        // make a trace array for each label
        var traces = [];

        if (isContinuousFloat(encoded_labels[1])) {
            // Create a single trace for continuous float labels
            traces.push({
                x: [],
                y: [],
                text: [],
                customdata: [],
                mode: 'markers',
                name: 'Continuous Data',
                marker: {
                    color: cliptoNinety(unpack(rows, keys[2])), size: 5, symbol: 'circle',
                    colorscale: 'Portland', showscale: true,
                    colorbar: { title: { text: keys[2] } }
                }
            });
        } else {
            encoded_labels[1].forEach(function (label, i) {
                if (keys[2] != dataset_en && encoded_dataset[1].length > 1) {
                    encoded_dataset[1].forEach(function (dataset, j) {
                        traces.push({
                            x: [],
                            y: [],
                            text: [],
                            customdata: [],
                            mode: 'markers',
                            name: `${label} - ${dataset}`,
                            marker: { color: label_color[label], size: 5, symbol: dataset_shapes[j], opacity: dataset_opacity[j] }
                        });
                    });
                } else {
                    traces.push({
                        x: [],
                        y: [],
                        text: [],
                        customdata: [],
                        mode: 'markers',
                        name: `${label}`,
                        marker: { color: label_color[label], size: 5, symbol: 'circle' }
                    });
                }
            });
        }

        // loop through the rows and append the data to the correct trace/data
        rows.forEach(function (row) {
            if (isContinuousFloat(encoded_labels[1])) {
                // Append data to the single trace for continuous float labels
                traces[0].x.push(row[keys[0]]);
                traces[0].y.push(row[keys[1]]);
                traces[0].text.push(row['ID']);
                traces[0].customdata.push(row['ID']);
            } else {
                // Existing logic for categorical labels
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

        // create the data array
        var data = traces;

        var layout = {
            dragmode: 'lasso',
            autosize: true,
            margin: {                           // update the left, bottom, right, top margin
                b: 20, r: 20, t: 20, l: 20
            },
            xaxis: { zeroline: false },
            yaxis: { zeroline: false },
            legend: {
                x: 1,
                //xanchor: 'right',
                //yanchor: 'top',
                y: 0.5
            },
            scene: { aspectmode: "cube", xaxis: { title: keys[0] }, yaxis: { title: keys[1] } }
        };

        // if there is only one trace, set the legend to false
        if (traces.length == 1) {
            layout.showlegend = false;

        }


        Plotly.react('graphDiv_scatter', data, layout, { responsive: true, });
        var graphDiv5 = document.getElementById("graphDiv_scatter")
        graphDiv5.on('plotly_selected', function (eventData) {
            var ids = []
            var ranges = []
            if (typeof eventData !== 'undefined') {
                eventData.points.forEach(function (pt) {
                    ids.push(pt.text);
                });
            } else if (eventData.length == 0) {
                console.log("No points selected, resetting table...");
                ids = undefined
            } else {
                console.log("No points selected, resetting table...");
                ids = undefined
            }
            filterByID(ids);
        });
        // graphDiv5.on('plotly_hover', function(data) {
        //     //update the hover innerHTML to display an image
        //     var point = data.points[0];
        //     var mouseX = data.event.clientX;
        //     var mouseY = data.event.clientY;
        //     var url = "./data/traces/" + point.text + ".svg"
        //     // Check if the hoverwrapper element exists
        //     var div = $('#hoverwrapper');
        //     if (div.length === 0) {
        //         // Create the hoverwrapper element if it does not exist
        //         div = $('<div id="hoverwrapper" class="image-wrapper card card-base">')
        //             .appendTo(document.body);
        //     }

        //     // Update the position and content of the hoverwrapper element
        //     div.css({
        //         "left": mouseX + 'px',
        //         "top": mouseY + 'px',
        //         "position": "absolute",
        //         "padding": "10px",
        //     }).html('<h6>'+ point.text+'</h6><img src="' + url + '" alt="Traces" style="max-width: 250px; max-height: 250px">');

        // })
    };

    //table functions
    function traceFormatter(index, row) {
        var html = []



        html.push('<div id="' + row.ID + '"></div>');

        html.push('</div>');

        setTimeout(() => { maketrace(row) }, 1000);
        return html.join('');
    }

    function plotFormatter(index, row) {

        return '<div id="' + row.ID + '"></div>';
    };


    function maketrace(row) {
        var url = "./data/traces/" + row.ID + ".svg"
        var html = []
        html.push('<img src="' + url + '" alt="Traces">');
        //get the div
        var div = document.getElementById("graphDiv_" + row.ID + "_plot")
        div.innerHTML = html.join('');

    };
    function makerheo(row) {
        var url = "./data/traces/" + row.ID + "_rheo.png"
        var html = []
        html.push('<img src="' + url + '" alt="Rheobase" style="width: 10%">');
        //get the div
        var div = document.getElementById("graphDiv_" + row.ID + "_plot_rheo")
        div.innerHTML = html.join('');
    };
    function makefi(row) {
        var url = "./data/traces/" + row.ID + "_FI.svg"
        var html = []
        html.push('<img src="' + url + '" alt="FI">');
        //get the div
        var div = document.getElementById("graphDiv_" + row.ID + "_plot_fi")
        div.innerHTML = html.join('');
    };

    function makeephys(row, keys = ekeys) {
        var html = [];

        //we just need to populate it with rows
        html.push('<div class="col table-ephys">');

        //loop through the keys
        keys.forEach(function (key) {
            html.push('<div class="row">');
            html.push('<span class="ephys-key">' + key + '</span>');
            html.push('<span class="ephys-value"> ' + row[key] + '</span>');
            html.push('</div>');


        });
        html.push('</div>');
        //get the div

        var div = document.getElementById("table_" + row.ID)
        div.innerHTML = html.join('');
    };

    function makeLink(row) {
        if (table_links.length > 0) {
            // Get the row ID
            var ID = row.ID;

            // Get the div
            var div = document.getElementById("link_" + ID);

            if (!div) {
                console.error(`Div with ID link_${ID} not found.`);
                return;
            }

            // Create a dropdown (select element)
            var parent_drop = document.createElement("div");
            parent_drop.className = "dropdown";

            var drop_button = document.createElement("button");
            drop_button.className = "btn btn-secondary dropdown-toggle";
            drop_button.type = "button";
            drop_button.id = "dropdownMenuButton" + ID;
            drop_button.setAttribute("data-bs-toggle", "dropdown");
            drop_button.setAttribute("aria-haspopup", "true");
            drop_button.setAttribute("aria-expanded", "false");
            drop_button.innerHTML = "Links";
            parent_drop.appendChild(drop_button);


            var select = document.createElement("div");
            select.className = "dropdown-menu";
            select.setAttribute("aria-labelledby", "dropdownMenuButton" + ID);

            // Add options for each link
            for (var i = 0; i < table_links.length; i++) {
                var link = table_links[i];
                var url = row[link];
                var option = document.createElement("a");
                option.className = "dropdown-item";
                option.text = link;
                option.href = url;
                select.appendChild(option);
            }

            // Clear the div and append the dropdown
            div.innerHTML = "";
            parent_drop.appendChild(select);
            div.appendChild(parent_drop);
        }
    };

    function filterByPlot(keys, ranges) {
        // check to see if the ranges are the same as the previous ranges, or within the bounds of the previous ranges
        var same = true;
        for (var i = 0; i < keys.length; i++) {
            if (prev_ranges[keys[i]] === undefined || ranges[i][0] < prev_ranges[keys[i]][0] || ranges[i][1] > prev_ranges[keys[i]][1]) {
                same = false;
                // update the prev_ranges
                prev_ranges
                break;
            }
        }
        // if the ranges are the same, do nothing
        if (same) {
            return;
        } else {
            prev_ranges = {};

            //we want to filter only the data selected on the scatter and parallel plots
            if (prev_filter != "scatter") { //if the previous filter was not the scatter plot, we want to filter by the parallel plot
                selected = []
            } else {
                var graphDiv_scatter = document.getElementById("graphDiv_scatter");
                var selected = []
                for (var i = 0; i < graphDiv_scatter.data.length; i++) {
                    //get the selected points
                    var trace = graphDiv_scatter.data[i];
                    var selectedIndices = trace.selectedpoints;
                    //if there are no selected points, skip this trace
                    if (selectedIndices === undefined) {
                        continue;
                    }

                    //get the IDs of the selected points
                    var selectedIDs = selectedIndices.map(function (value, index) {
                        return trace.text[value
                        ];
                    });
                    //update the selected array
                    selected.push(...selectedIDs);
                }
            };
        }
        //if the total number of selected points is 0, skip this step
        if (selected.length == 0) {
            var newArray = data_tb;
        } else {
            //filter the data_tb by the selected IDs
            var newArray = data_tb.filter(function (el) {
                return selected.includes(el.ID);
            })
        };
        //now we want to filter the data_tb by the selected ranges
        var newArray = newArray.filter(function (el) {
            return keys.every(function (key, i) {
                if (ranges[i][0] == -9999) {
                    return true;
                }
                else if (typeof ranges[i][0] === 'string') {
                    return ranges[i].includes(el[key]);
                }
                else {
                    return el[key] >= ranges[i][0] && el[key] <= ranges[i][1];
                }
            });
        });
        let result = newArray.map(function (a) { return a.ID; });

        $('#table').bootstrapTable('filterBy', { 'ID': result });
        crossfilter(data_tb, result, "parallel");
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
                    'background-color': 'hsla(0, 100%, 50%,' + (value / 40) + ')'
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
        $table.bootstrapTable('showLoading');
        var rows = $table.bootstrapTable('getData', { useCurrentPage: true }); // get the rows, only the visible ones\
        //determine if the visible rows are the same as the previous rows
        if (JSON.stringify(rows) === JSON.stringify(prev_rows)) {
            $table.bootstrapTable('hideLoading');
            return;
        } else {
            prev_rows = rows;
        }
        console.log("Generating plots...");
        let promises = rows.map(row => {
            return new Promise(resolve => {
                setTimeout(() => {
                    maketrace(row);
                    // Uncomment the next line if makerheo should also be awaited
                    // makerheo(row);
                    makeephys(row);
                    makefi(row);
                    makeLink(row);
                    resolve();
                }, 50);  // Reduced from 1000ms for faster rendering
            });
        });

        Promise.all(promises).then(() => {
            $table.bootstrapTable('hideLoading');
        });
    }

    function dataset_selector() {
        var selectedCheckboxes = document.querySelectorAll('input[name="dataset-select"]:checked');
        var selectedValues = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
        table_concatenator(selectedValues);
        //complete refresh
        var selected = $('input[name="label-select"]:checked').val();
        generate_umap(data_tb, ['Umap X', 'Umap Y', selected]);
        generate_paracoords(data_tb, para_keys, selected)
        $table.bootstrapTable('load', data_tb)
        //$table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
        $table.bootstrapTable('refresh')

    }


    // create the table with pagination for better performance
    // find the table div
    var $table = $('#table')
    // create the table with pagination enabled
    $table.bootstrapTable('load', data_tb)
    $table.bootstrapTable('refreshOptions', {
        pagination: true,
        pageSize: 50,
        pageList: [25, 50, 100, 200]
    })
    $table.bootstrapTable('refresh')

    /* onload */

    //find the elements of 
    var drop_parent = document.getElementById("umap-drop-menu");
    //this is a bootsrap select

    //add an event listener
    drop_parent.addEventListener('change', function (e) {
        console.log("Umap label changed...");
        var selected = $('input[name="label-select"]:checked').val();
        // if the selected has the class multi-select, then this element has a sibiling that holds the actual selected value
        if ($('input[name="label-select"]:checked')[0].classList.contains('multi-key')) {
            //step forward 2 in the DOM
            selected = $('input[name="label-select"]:checked')[0].nextElementSibling.children[0].value;
        }

        //check if selected is
        if (selected === split_strs) {
            var selectedCheckboxes = document.querySelectorAll('input[name="dataset-select"]:checked');
            var selectedValues = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
            pre_selected_datasets = selectedValues;
            // force select all the datasets
            var checkboxes = document.querySelectorAll('input[name="dataset-select"]');
            checkboxes.forEach(checkbox => checkbox.checked = true);
            dataset_selector();
        } else if (pre_selected_datasets.length > 0) {
            //we want to restore the preselected datasets
            pre_selected_datasets.forEach(function (dataset) {
                var checkbox = document.getElementById(dataset);
                checkbox.checked = true;
            });
            //uncheck any other checkboxes
            var checkboxes = document.querySelectorAll('input[name="dataset-select"]');
            checkboxes.forEach(checkbox => {
                if (!pre_selected_datasets.includes(checkbox.value)) {
                    checkbox.checked = false;
                }
            });
            //reset the pre_selected_datasets
            pre_selected_datasets = [];
            dataset_selector();
        } else {

            var keys = ['Umap X', 'Umap Y', selected]
            // Get the currently filtered data from the table to preserve filtering
            var filtered_data = $table.bootstrapTable('getData');
            generate_umap(filtered_data.length > 0 ? filtered_data : data_tb, keys);
            generate_paracoords(data_tb, paracoordskeys, selected)

        };
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
            dataset_selector();
        });

    }




    //add an event listener for table changes
    $table.on('refresh.bs.table', function (e, name, args) {
        console.log(e, name, args)
        //if its a click cell, we actually want to just ignore it
        generate_plots();
    });

    generate_plots();



});
