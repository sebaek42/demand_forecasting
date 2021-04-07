var express = require('express');
var router = express.Router();

router.get('/', function(req, res){
    res.render('home/index');
})

router.get('/forecast', function(req, res){
    res.render('home/forecast');
})

module.exports = router;