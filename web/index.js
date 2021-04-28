var express = require('express');
var app = express();

require('dotenv').config(); //dotenv라는 미들웨어있어서 설치하면 .env라는 파일을 사용할수 있게됨 .env는 환경변수 설정이 담긴 파일.

// DB
// mysql.connect();

// Settings
app.set('view engine', 'ejs'); // view engine으로 ejs를 쓰겠다..여러종류를 쓰고싶다면? view engine이란?
app.use(express.static(__dirname+'/public')) // __dirname 현재 디렉토리까지의 절대경로
app.use(express.json()); // json형식을 쓰겠다 json으로 res, req 소통
app.use(express.urlencoded({extended: true}));

app.use('/', require('./routes/home')); // /로 들어오면 ./routes/home으로 보내겠다

var port = 8888;
app.listen(port, function(){
    console.log('server on! http://localhost:'+port)
});
