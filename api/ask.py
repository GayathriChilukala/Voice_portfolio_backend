import ModelClient, { isUnexpected } from "@azure-rest/ai-inference";
import { AzureKeyCredential } from "@azure/core-auth";

export default async function handler(req, res) {
  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }

  try {
    const { question } = req.body;
    
    if (!question) {
      return res.status(400).json({ error: 'Question is required in request body' });
    }

    const token = process.env.GITHUB_TOKEN;
    const endpoint = "https://models.github.ai/inference";
    const modelName = "deepseek/DeepSeek-R1";

    if (!token) {
      return res.status(500).json({ error: 'GITHUB_TOKEN environment variable is not set' });
    }

    const client = ModelClient(
      endpoint,
      new AzureKeyCredential(token),
    );

    const response = await client.path("/chat/completions").post({
      body: {
        messages: [
          { role: "user", content: question }
        ],
        max_tokens: 1000,
        model: modelName
      }
    });

    if (isUnexpected(response)) {
      throw response.body.error;
    }

    const answer = response.body.choices[0].message.content;

    return res.status(200).json({ 
      question: question,
      answer: answer,
      model: modelName 
    });

  } catch (error) {
    console.error("Error:", error);
    return res.status(500).json({ 
      error: 'Failed to get response from model',
      details: error.message 
    });
  }
}
