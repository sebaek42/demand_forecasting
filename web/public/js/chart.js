
var regions = ["Etc", "South East Asia", "North East Asia", "America", "Oceania",
"Europe", "China", "Japan", "Middle East"];
// 지도 파이 차트  /////// 바 차트 분리해야됨
var regionChartData = [];
var colChartData = [];
var drawChart = function(processedData) {
    colChartData = processedData[0];
    regionChartData = processedData[1];
    google.charts.load('current', {packages: ['corechart', 'bar', 'geochart']});
    google.charts.setOnLoadCallback(drawBarChart);
    google.charts.setOnLoadCallback(drawMapChart);
    google.charts.setOnLoadCallback(drawPieChart);
    google.charts.setOnLoadCallback(drawRowChart);
}

function drawRowChart() {
    var data = new google.visualization.DataTable();
    data.addColumn("string", "Country");
    data.addColumn("number", "Passenger");
    data.addRows(regionChartData.length);
    for(var i = 0; i < regionChartData.length; ++i) {
        data.setCell(i, 0, regionChartData[i].region);
        data.setCell(i, 1, regionChartData[i].value);
    }
    var options = {
        chart: {
            title: regionChartData[0].date,
        },
        bars: 'horizontal'
    };
    var chart = new google.charts.Bar(document.getElementById("rowchart_values"));

    chart.draw(data, google.charts.Bar.convertOptions(options));
}

function drawPieChart() {
    var data = new google.visualization.DataTable();
    data.addColumn("string", "Country");
    data.addColumn("number", "Passenger");
    data.addRows(regionChartData.length);
    for(var i = 0; i < regionChartData.length; ++i) {
        data.setCell(i, 0, regionChartData[i].region);
        data.setCell(i, 1, regionChartData[i].value);
    }
    var options = {
        title: regionChartData[0].date,
        pieHole: 0.4,
    };

    var chart = new google.visualization.PieChart(document.getElementById("piechart_values"));
    chart.draw(data, options);
}

function drawMapChart() {
    // Create and populate a data table
    var data = new google.visualization.DataTable();
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
    console.log(type);
    var returnData = []
    for(var i=0;i<data.length;i++){
        if(data[i].region === region && data[i].type === type){
            returnData.push(data[i])
        }
    }
    return returnData[idx]
}
function getParameter(){
    return new Promise(function(resolve, reject){
        var paramNames = ['regionRadios', 'typeRadios', 'dateIndex'];
        var url = document.location.href;
        var qs = url.substring(url.indexOf('?') + 1).split("&");
        var result = {};
        for(var i=0;i<qs.length;i++){
            qs[i] = qs[i].split("=");
            result[qs[i][0]] = decodeURIComponent(qs[i][1]);
        }
        for(var i=0;i<paramNames.length;i++){
            if(result[paramNames[i]] == undefined){
                if(i !== 2){
                    result[paramNames[i]] = 'total';
                } else {
                    result[paramNames[i]] = 0;
                }
            }
        }
        resolve(result);
    })
}

function processData(params){
    return new Promise(function(resolve, reject){
        var colChartData = getColChartData(data, params.regionRadios, params.typeRadios);
        var regionChartData = []
        for(var i=0;i<regions.length;i++){
            regionChartData.push(getRegionChartData(data, regions[i],
            params.typeRadios, params.dateIndex));
        }
        resolve([colChartData, regionChartData]);
    })
}
