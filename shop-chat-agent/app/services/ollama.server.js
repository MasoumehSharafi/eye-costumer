const DEFAULT_SYSTEM_PROMPT = `
You are the EyeDropShop assistant.

Speak naturally, clearly, and concisely, like a helpful human assistant.

Important rules:
- Never diagnose a medical condition.
- Never claim that you can select the correct treatment.
- For personal symptoms, recommend consulting an eye-care professional first.
- Never invent product information, prices, availability, policies, or medical facts.
- Use only the trusted EyeDropShop Brain and Shopify information provided to you.
- Understand follow-up questions using the conversation history.
`;

async function sendOllamaRequest({
  baseUrl,
  model,
  systemPrompt,
  messages,
  temperature = 0.3,
}) {
  const response = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content: systemPrompt,
        },
        ...messages,
      ],
      stream: false,
      think: false,
      options: {
        temperature,
      },
    }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status}`);
  }

  const data = await response.json();
  const answer = data.message?.content?.trim();

  if (!answer) {
    throw new Error("Ollama returned an empty response");
  }

  return answer;
}

export function createOllamaService() {
  const baseUrl =
    process.env.OLLAMA_URL || "http://127.0.0.1:11434";

  const model =
    process.env.OLLAMA_MODEL || "qwen3.5:9b";

  return {
    async chat(messages, { trustedContext = "" } = {}) {
      const systemPrompt = trustedContext
        ? `${DEFAULT_SYSTEM_PROMPT}

TRUSTED CONTEXT:
${trustedContext}

Use this context to answer the customer naturally.
Do not mention internal tools, the Brain, MCP, or "trusted context."
Do not introduce facts that are absent from the context.
If the context contains medical safety advice, preserve its meaning and present it first.
If the context does not contain the answer, say that you cannot confirm it.`
        : DEFAULT_SYSTEM_PROMPT;

      return sendOllamaRequest({
        baseUrl,
        model,
        systemPrompt,
        messages,
        temperature: 0.3,
      });
    },

    async resolveProductQuery(messages, userMessage) {
      const systemPrompt = `
Rewrite the customer's latest request as a standalone Shopify product search query.

Use the conversation history only to resolve references such as:
- it
- that product
- this one
- the eye drops

Return only the product search query.
Do not answer the customer's question.
Do not include explanations.

Example:
Earlier conversation: Soothe Preservative-Free Allergy + Dry Eye
Latest request: How much is it?
Output: Soothe Preservative-Free Allergy + Dry Eye
`;

      return sendOllamaRequest({
        baseUrl,
        model,
        systemPrompt,
        messages: [
          ...messages,
          {
            role: "user",
            content: userMessage,
          },
        ],
        temperature: 0,
      });
    },
  };
}