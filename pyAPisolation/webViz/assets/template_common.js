// Common functions shared between static and dynamic webViz modes
// This file contains utility functions used by both template.js and template_dyn.js

// ============ Data Utility Functions ============

function unpack(rows, key) {
    return rows.map(function(row) { 
        return row[key]; 
    });
}

function isContinuousFloat(labels) {
    return labels.every(label => typeof label === 'number' || label === undefined || label === null);
}

function encode_labels(data, label) {
    var labels = data.map(function (a) { return a[label] });
    var unique_labels = [...new Set(labels)];
    var encoded_labels = labels.map(function (a) { return unique_labels.indexOf(a) });
    return [encoded_labels, unique_labels];
}

// ============ Table Formatting Functions ============

function valFormatter(value, row, index, field) {
    // Format values for display in table
    var html = []
    html.push('<div class="col">')
    html.push('<div class="feat-title">' + field + '</div>')
    html.push('<span class="feat-value">' + value + '</span>')
    html.push('</div>')
    return html.join('')
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

// ============ Filter Functions ============

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

function filterByPlot(keys, ranges){
    // check to see if the ranges are the same as the previous ranges, or within the bounds of the previous ranges
    var same = true;
    for (var i = 0; i < keys.length; i++) {
        if (prev_ranges[keys[i]] === undefined || ranges[i][0] < prev_ranges[keys[i]][0] || ranges[i][1] > prev_ranges[keys[i]][1]) {
            same = false;
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
                var selectedIDs = selectedIndices.map(function(value, index) {
                    return trace.text[value];
                });
                //update the selected array
                selected.push(...selectedIDs);
            }
        }
    }
    //if the total number of selected points is 0, skip this step
    if (selected.length == 0) {
        var newArray = data_tb;
    } else {
        //filter the data_tb by the selected IDs
        var newArray = data_tb.filter(function (el) {
            return selected.includes(el.ID);
        })
    }
    //now we want to filter the data_tb by the selected ranges
    var newArray = newArray.filter(function (el) {
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
}

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
        prev_filter = "parallel";
    } else if (sender == "scatter") {
        //in this case we completely reset the parallel plot
        generate_paracoords(data_tb, paracoordskeys, paracoordscolors, IDs);
        prev_filter = "scatter";
    } else {
        //do nothing
    }
    //set the restyle flag to false
    restyle_programmatically = false
}

// ============ Table Concatenation ============

function table_concatenator(labels){
    //update the global table data_tb with the selected labels
    data_tb = []
    labels.forEach(function(label) {
        // Assuming you have a way to get data for each label
        var dataForLabel = subtables[label]
        // Concatenate or merge dataForLabel into data_tb
        data_tb.push(...dataForLabel);
    });
}

// ============ Electrophysiology Data Display ============

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
    if (div) {
        div.innerHTML = html.join('');
    }
}

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
}

// ============ Dataset Selector ============

function dataset_selector(){
    var selectedCheckboxes = document.querySelectorAll('input[name="dataset-select"]:checked');
    var selectedValues = Array.from(selectedCheckboxes).map(checkbox => checkbox.value);
    table_concatenator(selectedValues);
    //complete refresh
    var selected = $('input[name="label-select"]:checked').val();
    generate_umap(data_tb, ['Umap X', 'Umap Y', selected]);
    generate_paracoords(data_tb, para_keys, selected)
    $table.bootstrapTable('load', data_tb)
    $table.bootstrapTable('refresh')
}
