var express = require('express');
var router = express.Router();
const axios = require("axios");

router.get('/', function(req, res, next) {
  res.render("user/testpage")
  return
});

router.post("/get_news", async (req, res) => {
  try {
      const text = req.body.news_content;
      console.log("TEXT : ",text)
      const response = await axios.post("http://127.0.0.1:8000/predict/", 
          { text: text }, // Ensure it's sent as JSON
          { headers: { "Content-Type": "application/json" } } // Set content type
      );
      console.log("RESPONSE : ",response)
      res.json(response.data);
  } catch (error) {
      console.error("Error: ", error.response ? error.response.data : error.message);
      res.status(500).json({ error: "Something went wrong" });
  }
});

module.exports = router;

