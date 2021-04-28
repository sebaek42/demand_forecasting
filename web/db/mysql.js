var mysql = require('mysql2');
require('dotenv').config();

var pool = mysql.createPool({
    host: process.env.MYSQL_HOST,
    user: process.env.MYSQL_USER,
    password: process.env.MYSQL_PASSWORD,
    database: process.env.MYSQL_DB,
    dateStrings: 'date'
});
const promisePool = pool.promise();//얘도 프로미스..객체를 사용하기위함?
module.exports = promisePool;
