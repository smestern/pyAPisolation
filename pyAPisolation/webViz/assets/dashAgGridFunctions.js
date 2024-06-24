var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});


// function to make a button that exapnds the row, using react
dagcomponentfuncs.expandRow = function (params) {
    const {setData, data} = params;
    var eButton = React.createElement('button', {
        onClick: function () {
            // we dont have enterprise AG Grid, so we can't use the gridApi, so we will use the rowNode and brute force it
            var rowNode = params.node;
            //rowNode.setRowHeight(500);//
            //spawn a new row node with the data
            

            // however we dont want to expand the data columns, so we will set the data column to remain their original height
            params.api.onRowHeightChanged();
            //now we want to spawn a inner row containing a dcc graph
            // we will use the rowNode id to create a new div
            var newDiv = document.createElement('div');
            newDiv.id = 'innerRow' + rowNode.id;
            newDiv.style.height = '500px';
            newDiv.style.width = '100%';
            newDiv.style.backgroundColor = 'white';
            newDiv.style.border = '1px solid black';
            newDiv.style.margin = '10px';
            newDiv.style.padding = '10px';
            newDiv.style.display = 'flex';
            newDiv.style.flexDirection = 'row';
            newDiv.style.justifyContent = 'left';
            newDiv.style.alignItems = 'left';
            setData(data);
            //add the data as a string to the div
            newDiv.innerHTML = data;
            
            
            //eDiv.appendChild(newDiv);


            // create a function to listen for new divs under the dom 'datatable-row-ids-store'
            // this will be used to listen for the data of the graph being dumped into the div
            var data_store = document.getElementById('datatable-row-ids-store');
            // create an observer instance
            var observer = new MutationObserver(function(mutations) {
                [mutations[0]].forEach(function(mutation) {
                    
                    //get the children of the data store
                    var children = mutation.target.children;
                    //iterate through the children
                    for (var i = 0; i < children.length; i++) {
                        //get the id of the child
                        var id = children[i].id;
                        //check if the id is the same as the newDiv
                        if (id == 'graph' + rowNode.id) {
                            //if it is, then we will remove the observer and move the child to the new div
                            //remove the observer
                            observer.disconnect();
                            //get the new div
                            var newDiv = document.getElementById('innerRow' + rowNode.id);
                            //get the child
                            var child = children[i];
                            //read the text of the child as a json
                            var data = JSON.parse(child.innerText);
                            //create a new graph
                            var newGraph = React.createElement(window.dash_core_components.Graph, {figure: data,
                                style: {height: '100%'},
                                config: {displayModeBar: false},
                            })
                            //render the graph in the new div
                            //ReactDOM.render(newGraph, newDiv);
                            params.api.applyTransaction({add: [{data: [], id: 'graph' + rowNode.id, type: 'graph'}], addIndex: params.node.rowIndex+1});

                            //get the inserted row
                            var insertedRow = params.api.getDisplayedRowAtIndex(params.node.rowIndex + 1);

                            //place the graph in the inserted row
                            // now we will append the new div to the row node
                             
                            var eDiv = params.api.getRowNode(insertedRow.id);
                            eDiv.appendChild(newGraph);



                        }
                    }
                });
            });
            // configuration of the observer:
            var config = { attributes: true, childList: true, characterData: true };
            // pass in the target node, as well as the observer options
            observer.observe(data_store, config);
            
            
        }
    }, 'Expand'); 
    return eButton;
}

