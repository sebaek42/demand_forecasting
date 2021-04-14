const e = require('express');
var express = require('express');
var mysql = require('../db/mysql');
var router = express.Router();

router.get('/', function(req, res){
    res.render('home/index');
})

router.get('/forecast', function(req, res){
    res.render('home/forecast', {forecast:{}});
})

router.get('/forecast/show', function(req,res){
    var region = req.query.regionRadios
    var type = req.query.typeRadios
    getData(region, type)
    .then(function(data){
        res.locals.forecast = data;
        res.render('home/forecast');
    });
})

module.exports = router;

function getData(region, type){
    var sql = "SELECT * FROM forecasted_data WHERE region=? and type=?";
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