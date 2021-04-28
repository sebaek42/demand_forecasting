<<<<<<< HEAD
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
=======
const e = require('express');
var express = require('express');
var mysql = require('../db/mysql');
var router = express.Router();

router.get('/', function(req, res){
    res.render('home/index'); // express에서 기본적으로 명시를 안해줘도 경로가 views/디렉토리로 시작된다. 아까 기본적으로 viewengine으로 ejs를 설정했기때문에 .ejs생략가능
})

router.get('/forecast', function(req, res){
    res.render('home/forecast', {forecast:{}}); // forecast:{} 빈데이터를 보내주겠다는거
})

router.get('/forecast/show', function(req,res){ // 콜백함수는 내부의 함수를 끝날때까지 기다리지않는 개 양아치 함수임
    var region = req.query.regionRadios
    var type = req.query.typeRadios
    getData(region, type) // 그래서 겟데이터가 미처 자기 작업을 끝내기 전에 콜백함수가 끝나버릴수있음. 그러면 html파일에 데이터가 전달되지 않은 상태가되버린다
    .then(function(data){ // 그래서 비동기처리를 해줘야하는거고 promise객체에 대한 공부가 필요해지느거. .then으느 그걸 위해쓰임.
        res.locals.forecast = data;
        res.render('home/forecast');
    });
})

module.exports = router;

function getData(region, type){
    var sql = "SELECT * FROM forecasted_data WHERE region=? and type=?";
    return mysql.query(sql, [region, type]).then(parsingData);
}

async function parsingData(rows){ // 예약어 async를 주면 밑에 await 들어오는 키워드 뒤의 함수를 기다렸다가 이후의 코드 진행하겠다. 비동기 처리
    var results = rows[0];
    var returnArr = [];
    for(var i=0;i<results.length;i++){
        var data = results[i];
        var date = await parsingDate(data['date']); // 뭘 기다릴거냐가
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
>>>>>>> cfc3ff71d7ee49b8b6d3b268df9f393ca070d928
