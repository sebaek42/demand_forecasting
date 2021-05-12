var express = require('express');
var mysql = require('../db/mysql');
var request = require('sync-request');
var convert = require('xml-js');
var router = express.Router();

router.get('/', function(req, res){
    res.render('home/index');
})

router.get('/forecast', function(req, res){
    var sql = "SELECT * FROM forecasted_data";
    getData(sql)
    .then(function(data){
        return new Promise(function(resolve, reject){
            var weights = getWeights();
            for(var i=0;i<data.length;i++){
                data[i].value = parseInt(data[i].value*weights[data[i].region]);
                weights[data[i].region] -= 0.015;
            }
            console.log(data);
            resolve(data);
        })
    }).then(function(data){
        res.render('home/forecast' , { forecast: JSON.stringify(data) })
    });
})
module.exports = router;

function getData(sql, region, type){
    return mysql.query(sql, [region, type]).then(parsingData);
}

async function parsingData(rows){
    var results = rows[0];
    var returnArr = [];
    for(var i=0;i<results.length;i++){
        var data = results[i];
        var date = await parsingDate(data['date']);
        var value = data.value;
        var type = data.type;
        var region = data.region;
        returnArr.push({type, region, value, date});
    }
    return returnArr;
}

function parsingDate(date){
    var splitDate = `${date}`.split('-');
    var year = splitDate[0];
    var month = splitDate[1];
    return `${year}-${month}`;
}

function getWeights() {
    var countryIsoCode = ['ISR','USA','JPN','CHN','HKG','THA','AUS','FRA']
    var url = 'http://apis.data.go.kr/1262000/TravelSpecialWarningService/getTravelSpecialWarningList';
    var queryParams = '?' + encodeURIComponent('ServiceKey') + `=${process.env.API_KEY}`; /* Service Key*/
    for(var i=1;i<countryIsoCode.length+1;i++){
        queryParams += '&' + encodeURIComponent(`isoCode${i}`) + '=' + encodeURIComponent(countryIsoCode[i-1]);    
    }
    var res = request("GET", url + queryParams);
    var body = res.getBody('utf8');
    var json = JSON.parse(convert.xml2json(body, {compact:true, spaces:4}));
    var items = json.response.body.items.item;
    return dangerToNumeric(items);
    
}
function dangerToNumeric(items) {
    var IsoToContinent = {
        'ISR': 'Middle East',
        'USA': 'America',
        'JPN': 'Japan',
        'CHN': 'China',
        'HKG': 'North East Asia',
        'THA': 'South East Asia',
        'AUS': 'Oceania',
        'FRA': 'Europe'
    }
    var sum = 1;
    var tripDanger = {'Etc': 1};
    for(var i in items){
        var isoCode = items[i].isoCode._text; 
        var continent = IsoToContinent[isoCode];
        if(items[i].splimit !== undefined){
            tripDanger[continent] = 0.95;
        } else if(items[i].splimitPartial !== undefined){
            tripDanger[continent] = 0.96;
        } else if(items[i].spbanYna !== undefined){
            tripDanger[continent] = 0.98;
        } else if(items[i].spbanYnPartial !== undefined){
            tripDanger[continent] = 0.97;
        } else {
            tripDanger[continent] = 1;
        }
        sum += tripDanger[continent]; 
    }
    tripDanger['total'] = parseFloat((sum / Object.keys(tripDanger).length).toFixed(2));
    return tripDanger;
}
