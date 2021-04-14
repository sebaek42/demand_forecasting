var express = require('express');
const { query } = require('./db/mysql');
// var mysql = require('./db/mysql');
var app = express();

require('dotenv').config();

// DB
// mysql.connect();

// Settings
app.set('view engine', 'ejs');
app.use(express.static(__dirname+'/public'))
app.use(express.json());
app.use(express.urlencoded({extended: true}));

app.use('/', require('./routes/home'));

var port = 8888;
app.listen(port, function(){
    console.log('server on! http://localhost:'+port)
});
