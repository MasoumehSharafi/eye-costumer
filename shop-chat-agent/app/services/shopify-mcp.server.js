const AGENT_PROFILE =
  "https://shopify.dev/ucp/agent-profiles/examples/2026-08-25/valid-with-capabilities.json";

const STOREFRONT_DOMAIN = (
  process.env.SHOPIFY_STOREFRONT_DOMAIN ||
  "eye-agent-test.myshopify.com"
)
  .replace(/^https?:\/\//, "")
  .replace(/\/+$/, "");

async function callShopifyMcp(endpoint, name, args) {
  const response = await fetch(
    `https://${STOREFRONT_DOMAIN}${endpoint}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: Date.now(),
        method: "tools/call",
        params: {
          name,
          arguments: args,
        },
      }),
    },
  );

  if (!response.ok) {
    throw new Error(
      `Shopify MCP error ${response.status} for ${name}`,
    );
  }

  const payload = await response.json();

  if (payload.error) {
    throw new Error(
      payload.error.message || "Shopify MCP request failed",
    );
  }

  return payload.result?.structuredContent || {};
}

export function createShopifyMcpService() {
  return {
    async searchCatalog(query) {
      const result = await callShopifyMcp(
        "/api/ucp/mcp",
        "search_catalog",
        {
          meta: {
            "ucp-agent": {
              profile: AGENT_PROFILE,
            },
          },
          catalog: {
            query,
            context: {
              address_country: "CA",
              language: "en",
              currency: "CAD",
            },
            pagination: {
              limit: 3,
            },
          },
        },
      );

      return result.products ? result : {
        products: [],
      };
    },

    async searchPolicies(query, context = "") {
      return callShopifyMcp(
        "/api/mcp",
        "search_shop_policies_and_faqs",
        {
          query,
          ...(context ? { context } : {}),
        },
      );
    },
  };
}
