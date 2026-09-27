# EyeDropShop Customer Agent

Customer-facing Shopify assistant backed by the EyeDropShop Brain and live edsAI product data.

## Project structure

- `eyedropshop-brain/`: Python/FastAPI query understanding, edsAI retrieval, grounded response generation, safety routing, evaluation, and audit logic.
- `shop-chat-agent/`: Shopify app proxy, signed conversation handling, chat UI, and Brain API client.
- `start-demo.sh`: local launcher for the Brain and Shopify development server.
- `CUSTOMER_AGENT_DEBUG_REPORT.md`: implementation and validation notes.

## Local setup

1. Copy `.env.example` values into local `.env` files as appropriate. Never commit those files.
2. Install the Python dependencies from `eyedropshop-brain/requirements.txt`.
3. Install the Node dependencies in `shop-chat-agent/` with `npm install`.
4. Configure the Shopify app and edsAI renewal credential.
5. Run `./start-demo.sh`.

The repository intentionally excludes API keys, tokens, local databases, private certificates, logs, caches, build output, and dependency directories.

