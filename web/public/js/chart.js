
var eachData = {}
var drawBarChart = function(data) {
    google.charts.load('current', {packages: ['corechart', 'bar']});
    google.charts.setOnLoadCallback(drawAnnotations);
    eachData = data;
}

function drawAnnotations() {
    if(eachData.length > 0){
        var dateAndValue = []
        for(var i=0;i<eachData.length;i++){
            dateAndValue.push([eachData[i].date, eachData[i].value])
        }
        var data = google.visualization.arrayToDataTable([
        [ {label: 'Date', id: 'date'},
        {label: 'Passenger', id: 'passenger', type:'number'}],
        dateAndValue[0],
          dateAndValue[1],
          dateAndValue[2],
          dateAndValue[3],
          dateAndValue[4],
          dateAndValue[5],
        ]);
        
        var view = new google.visualization.DataView(data);
        var options = {
            title: `Forecasted Passenger - Type: ${eachData[0].type} / Region: ${eachData[0].region}`,
            width: '100%',
            height: 475,
            bar: {groupWidth: "95%"},
            legend: { position: "right" },

        };
        var chart = new google.visualization.ColumnChart(document.getElementById("columnchart_values"));
        chart.draw(view, options);
    } else {
       var columChart =  document.getElementById("columnchart_values");
       columChart.innerHTML = "<h2>Try Again</h2>";
    }
}