const compression = require("compression");
const express = require("express"); // create an express app
const app = express();
const port = 5000;
const { GoogleGenerativeAI } = require("@google/generative-ai");
app.use(compression());

require("dotenv").config();
const cors = require("cors");
const path = require("path");
const fs = require("fs");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Read the personal info from the sample_text.txt file
const personalInfo = fs.readFileSync('./sample_text.txt', 'utf-8');

const chat = model.startChat((history = []));

// route for chatbot interactions
app.post("/chat", async (req, res) => {
  const { message } = req.body;

  const prompt = `
You are a knowledgeable and helpful AI chatbot designed to answer questions about B. A. Akith Chandinu, an undergraduate from the University of Moratuwa, Faculty of IT. You specialize in providing clear, accurate, and informative answers based on the following details:

**Background**:
${personalInfo}

With this information, answer the following question in a friendly, detailed, and accurate manner. Give short answers if possible: "${message}"
`;

  try {
    const result = await chat.sendMessage(prompt);

    res.json(result.response.text());

    console.log("Response generated successfully.");
  } catch (error) {
    console.error("Error generating response:", error);
    res.status(500).send("Error processing request.");
  }
});

app.get("/", (req, res) => {
  // this is an endpoint
  res.send("The Backend Server is running!");
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}/`);
});
