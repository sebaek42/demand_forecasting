var express = require('express');
var mysql = require('mysql');
var router = express.Router();
require('dotenv').config();

var connection = mysql.createConnection({
    host: process.env.MYSQL_HOST,
    user: process.env.MYSQL_USER,
    password: process.env.MYSQL_PASSWORD,
    database: process.env.MYSQL_DB
});


router.get('/', function(req, res){
    res.render('home/index');
})

router.get('/forecast', function(req, res){
    res.render('home/forecast');
})

router.get('/forecast/show', function(req,res){
    var region = req.query.regionRadios
    var type = req.query.typeRadios
    var data = getData(region, type);
})

module.exports = router;

function getData(region, type){
    connection.connect();
    var sql = '';
    if(region == "total"){
        sql = `SELECT * FROM passenger_data WEHRE type=${type}`;
    } else {
        sql = `SELECT * FROM region_data WEHRE region=${type} and type=${type}`;
    }
    connection.query("SELECT * FROM passenger_data", function(err, results, fields){
        if(err) return err;
        console.log(results);
    })
}