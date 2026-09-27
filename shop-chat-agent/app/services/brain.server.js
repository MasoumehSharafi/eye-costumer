export function createBrainService() {
  const apiUrl =
    process.env.BRAIN_API_URL || "http://127.0.0.1:8000";

  const apiKey = process.env.BRAIN_CUSTOMER_API_KEY;

  if (!apiKey) {
    throw new Error("BRAIN_CUSTOMER_API_KEY is missing");
  }

  return {
    async answer(question, language = "en", history = []) {
      const response = await fetch(`${apiUrl}/v1/answer`, {
        method: "POST",
        signal: AbortSignal.timeout(Number(process.env.BRAIN_REQUEST_TIMEOUT_MS || 240000)),
        headers: {
          "Content-Type": "application/json",
          "X-Brain-Role": "customer",
          "X-Brain-Key": apiKey,
        },
        body: JSON.stringify({
          question,
          language,
          history,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Brain API error ${response.status}`,
        );
      }

      return response.json();
    },
  };
}