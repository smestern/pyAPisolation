// This script is used to generate the table and parallel coordinates plot in the web application
// wrap this in a on load function to ensure the page is loaded before the script runs
window.onload = function() {

    /* data_tb */

    var embed_colors = ['#0000FF', '#A5E41F', '#FF24FF', '#B8B2B2', '#fc0303']
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
                values: unpack(data_tb, key),
                multiselect: false
            }}),
            rangefont: {size: 5},
            labelangle: -45
        }]; // create the data object
        
        var layout = {
        };
        
        fig = Plotly.newPlot('graphDiv_parallel', data, layout, {responsive: true}); // create the plots
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
                        if (allLengths.length > 2){
                            ranges.push([d.constraintrange[0][0],d.constraintrange[0][1]]); //return only the first filter applied per feature

                        }else{
                            ranges.push(d.constraintrange);
                        }
                        
                        
                    } // => use this to find values are selected
            })

            filterByPlot(keys, ranges)
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
        // make a trace array for each label
        var traces = []
        encoded_labels[1].forEach(function (label, i) {traces.push(new Object())});

        // loop through the rows and append the data to the correct trace/data
        rows.forEach(function (row) {
            var trace = encoded_labels[1].indexOf(row[keys[2]]);
            if (traces[trace].x === undefined) {
                traces[trace].x = [];
                traces[trace].y = [];
                traces[trace].text = [];
                traces[trace].name = encoded_labels[1][trace];
                traces[trace].mode = 'markers';
                traces[trace].marker = { color: colors[trace], size: 5 };
            }
            traces[trace].x.push(row[keys[0]]);
            traces[trace].y.push(row[keys[1]]);
            traces[trace].text.push(row['ID']);
        });

        // create the data array
        var data = traces;

        var layout = {dragmode: 'lasso',autosize: true,
            margin: {                           // update the left, bottom, right, top margin
                b: 20, r: 10, t: 20
            },};

        Plotly.react('graphDiv_scatter', data, layout, { responsive: true });
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
        html.push('<object data="' + url + '" alt="Traces" style="width: 30vw">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot")
        div.innerHTML = html.join('');
        
    };
    function makerheo(row){
        var url = "./data/traces/" + row.ID + "_rheo.png"
        var html = []
        html.push('<img src="' + url + '" alt="Rheobase" style="width: 10vw">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot_rheo")
        div.innerHTML = html.join('');
    };
    function makefi(row){
        var url = "./data/traces/" + row.ID + "_FI.svg"
        var html = []
        html.push('<object data="' + url + '" alt="FI" style="width: 20vw">');
        //get the div
        var div = document.getElementById("graphDiv_"+row.ID+"_plot_fi")
        div.innerHTML = html.join('');
    };

    function filterByPlot(keys, ranges){		
        var newArray = data_tb.filter(function (el) {
                return keys.every(function (key, i) {
                    if (ranges[i][0] == -9999){
                        return true;
                    }
                    else{
                        return el[key] >= ranges[i][0] && el[key] <= ranges[i][1];
                    }
                });	
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
    // create the table
    $table.bootstrapTable('load', data_tb)
    // while we are here, set the attr 'data-detail-formatter' to the function we defined above

    //add an event listener for table changes
    $table.on('all.bs.table', function (e, name, args) {
        generate_plots();
    });

    generate_plots();

    // refresh the table
    // set the table to be responsive
    //$table.bootstrapTable('refreshOptions', {detailView: true, detailFormatter : traceFormatter})
    $table.bootstrapTable('refresh')

    //now create our cell plots
    

};
