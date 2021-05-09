
var regions = ["Etc", "South East Asia", "North East Asia", "America", "Oceania",
"Europe", "China", "Japan", "Middle East"];
// 지도 파이 차트  /////// 바 차트 분리해야됨
var drawChart = function() {
    google.charts.load('current', {packages: ['corechart', 'bar', 'geochart']});
    google.charts.setOnLoadCallback(drawBarChart);
    google.charts.setOnLoadCallback(drawMapChart);
}
function drawMapChart() {
    // Create and populate a data table
    var data = new google.visualization.DataTable();
    var regions = ["Etc", "South East Asia", "North East Asia", "America", "Oceania",
    "Europe", "China", "Japan", "Middle East"]
    data.addColumn("string", "Country");
    data.addColumn("number", "Passenger");
    data.addRows(regionChartData.length);
    for(var i=0;i<regionChartData.length;i++){
        data.setCell(i, 0, regionChartData[i].region);
        data.setCell(i, 1, regionChartData[i].value);
    }
    // Instantiate our Geochart GeoJSON object
    var vis = new geochart_geojson.GeoChart(document.getElementById("mapchart_values"));
    // Set Geochart GeoJSON options
    var options = {
        'title': regionChartData[0].date,
        mapsOptions: {
        center: {lat: 50, lng: 70},
        zoom: 1.85
        },
        geoJson: "https://ghcdn.rawgit.org/sebaek42/demand_forecasting/main/test/world.js",
        geoJsonOptions: {
        idPropertyName: "COUNTRYAFF"
        }
    };
    // Draw our Geochart GeoJSON with the data we created locally
    vis.draw(data, options);
}
function drawBarChart() {
    var data = new google.visualization.DataTable();
    data.addColumn('string', 'Date');
    data.addColumn('number', 'Passenger');
    data.addRows(colChartData.length);
    for(var i=0;i<colChartData.length;i++){
        data.setCell(i, 0, colChartData[i].date);
        data.setCell(i, 1, colChartData[i].value);
    }
    
    var view = new google.visualization.DataView(data);
    var options = {
        title: `Forecasted Passenger - Type: ${colChartData[0].type} / Region: ${colChartData[0].region}`,
        width: '100%',
        height: 475,
        bar: {groupWidth: "95%"},
        legend: { position: "right" },

    };
    var chart = new google.visualization.ColumnChart(document.getElementById("columnchart_values"));
    chart.draw(view, options);
}

function getColChartData(data, region, type){
    var returnData = []
    for(var i=0;i<data.length;i++){
        if(data[i].region === region && data[i].type === type){
            returnData.push(data[i])
        }
    }
    return returnData;
}

function getRegionChartData(data, region, type, idx){
    var returnData = []
    for(var i=0;i<data.length;i++){
        if(data[i].region === region && data[i].type === type){
            returnData.push(data[i])
        }
    }
    return returnData[idx]
}