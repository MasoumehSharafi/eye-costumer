# Customer Agent debugging and verification

## Request flow and root cause

Theme `chat-interface.liquid` loads `chat.js`. The corrected browser endpoint is
`/apps/eye-agent` (same origin), which Shopify forwards to the app's `/chat` route.
React Router verifies Shopify's proxy signature, validates the signed conversation
ID, loads database history, and calls Brain `/v1/answer` with the server-only customer
key. Brain resolves references, retrieves edsAI evidence, generates an answer, and
records evidence/citations and a trace ID. The storefront receives display text via SSE.

The linked config is `shop-chat-agent/shopify.app.shop-chat-agent.toml`.
The live CLI confirmed the existing app `shop-chat-agent` and development store
`eye-agent-test.myshopify.com`. Before these edits, that config had neither
`write_app_proxy` nor `[app_proxy]`. The unlinked `shopify.app.toml` had the proxy
configuration but still has a placeholder client ID; it is not the config to run.

A second routing bug appended `/chat` in the browser despite the proxy target
already being `/chat`: `/apps/eye-agent/chat` would forward to `/chat/chat`, which
is not the intended route. GET history had the same error. The generic browser
error is emitted when fetch fails or returns an unsuccessful response. These are
confirmed code/config defects; the original failing browser request's status was
not captured, so no historical HTTP status is claimed.

No second app was created. No manual tunnel hostname is needed in the theme.
The existing Liquid file already supplied the stable proxy path and needed no edit.

## Files changed

| File | Reason |
| --- | --- |
| `start-demo.sh` | Pin the linked app/config and store; allow an alternate theme-preview port through `SHOPIFY_THEME_PORT`; remove unused asset variable. |
| `shop-chat-agent/.gitignore` | Include the actual linked app config in version control. |
| `shop-chat-agent/shopify.app.shop-chat-agent.toml` | Add `write_app_proxy` and `/apps/eye-agent` → `/chat`. |
| `shop-chat-agent/app/routes.js` | Exclude `.backup` files; the build previously registered `chat.jsx.backup` as another live route. |
| `shop-chat-agent/app/routes/chat.jsx` | Authenticate proxy requests; strip display citation markers after Brain auditing; log trace/status and a controlled dependency error. |
| `shop-chat-agent/app/services/brain.server.js` | Avoid embedding upstream response bodies in thrown errors; retain authenticated customer calls and bounded history. |
| `shop-chat-agent/extensions/chat-bubble/assets/chat.js` | Fix POST/history URLs, check SSE content type, await the stream, hide markers in older displayed history. |
| `eyedropshop-brain/edsai_retrieval.py` | Retrieve description/details plus ingredients for preservative questions; let alternative searches proceed to semantic retrieval. |
| `eyedropshop-brain/edsai_intelligence.py` | Clarify reference-product handling for alternative searches in the LLM prompt. |
| `eyedropshop-brain/answer_generator.py` | Fix price/inventory field parsers; retain lookup citations; prevent ingredient-role inference without explicit evidence; limit overview sentences; keep counts out of customer generation; preserve warning detail; exclude customer CSV ambiguity and hard-coded support contact fallbacks. |
| `eyedropshop-brain/tests/test_customer_upgrade.py` | Add greeting, preservative evidence, overview, currency/price, and inventory regressions. |
| `shop-chat-agent/tests/customer-chat.test.mjs` | Add proxy auth boundary, citation display, and URL/query regressions. |
| `CUSTOMER_AGENT_DEBUG_REPORT.md` | Record results, restart commands, and acceptance tests. |

The existing resolver, signed-token format, session secret, database history, and
Brain customer authentication are retained. Doctor/Internal retrieval remains
available. Customer evidence is additionally filtered to edsAI at generation entry.

## Evidence and quality findings

Live edsAI LUMIFY records include ingredients, usage, warnings, description, and
multiple priced variants. The ingredient list includes benzalkonium chloride but
the inspected fields do not explicitly establish its preservative role. An initial
live model run made that inference anyway. The new guard returns the source's
ingredient list and states that preservative status is unspecified instead.

Initial live generation also exceeded the overview sentence instruction and exposed
inventory counts for multiple variants. Final code limits description-only customer
overviews to three complete sentences and transforms counts into availability
statements before customer generation. Original inventory evidence stays in the
Brain response/audit pipeline. Warning/usage/comparison answers are not sentence-trimmed.

Only edsAI numeric prices are used; no currency symbol/code is added. Unknown or
untracked inventory cannot establish availability. Alternative discovery returned
no adequate evidence in the live test; it did not assert a preservative-free equivalent.

## Verification completed and limits

- 17 Python offline regressions passed.
- 10 Node route/service/client regressions passed. Route tests mock the Shopify
  authenticator; they verify that its boundary is called, not a real Shopify signature.
- React Router production build passed, with the backup route excluded.
- Browser JavaScript and launcher syntax checks passed; repository diff checks passed.
- Live CLI confirmed the intended existing app/config/store, granted
  `unauthenticated_read_product_listings, write_app_proxy`, and reported a tunnel
  proxy target ending in `/chat`.
- All eight requested turns returned HTTP 200 through the authenticated live Brain.
  Product facts cited only edsAI. Follow-ups preserved LUMIFY context.
- A fresh live test verified the final preservative guard. Final overview limiting
  and inventory redaction were verified by regression tests after the initial live
  run exposed those problems; the full eight-turn run was not repeated afterward.
- Storefront POST returned HTTP 302 to `/password`. No browser was available to
  unlock the store, so browser-through-Shopify POST/history/SSE delivery is not
  claimed verified. A direct signed-tunnel probe could not run because Shopify's
  signing secret was not present in `.env` (CLI injects app credentials at runtime).
- The test demo processes were stopped; the pre-existing development process and
  Ollama were left alone. Restart to load the final Python changes.

## Restart and tests

Stop the old demo with Control+C in its terminal first. Then:

```sh
cd ~/Documents/EyeDropShop/Costumer_Agent
zsh ./start-demo.sh
```

If another theme-preview process still occupies port 9293:

```sh
SHOPIFY_THEME_PORT=9294 zsh ./start-demo.sh
```

The launcher executes the existing app explicitly:

```sh
conda run --no-capture-output -n shop-chat-agent \
  shopify app dev --config shop-chat-agent \
  --store eye-agent-test.myshopify.com --theme-app-extension-port 9293
```

Use the launcher for normal startup; the standalone command above does not set its
Brain key/session environment. No `--reset`, app creation, or production deployment
is needed for this development test.

Run checks:

```sh
cd ~/Documents/EyeDropShop/Costumer_Agent
conda run -n eyedropshop-brain python -m unittest discover -s eyedropshop-brain/tests
cd shop-chat-agent
conda run -n shop-chat-agent node --test tests/customer-chat.test.mjs
conda run -n shop-chat-agent npm run build
```

Open the actual unlocked `https://eye-agent-test.myshopify.com` storefront with
the chat embed enabled and the CLI development preview active. If the installed
app's proxy prefix/subpath was previously customized, check its App proxy URL in
Shopify Admin: CLI explicitly notes prefix/subpath changes apply only to new installs.
Use `/apps/eye-agent` for this app; do not create another app to fix an existing setting.

## End-to-end acceptance conversation

Use one conversation, in this order. Expected behavior, not fabricated fixed answers:

| Customer message | Expected result |
| --- | --- |
| `hello` | Short greeting. |
| `hellp` | Greeting, not an evidence fallback. |
| `what is lumify?` | 2–3 concise sentences from edsAI. |
| `how much is it?` | LUMIFY variant prices, without invented currency. |
| `does it have preservatives?` | Explicit source statement if available; otherwise unspecified status and the actual ingredient list, without classifying ingredients using outside knowledge. |
| `any warnings?` | LUMIFY warnings, grouped/readable without removing important contraindications or safety limits. |
| `is it available?` | Availability per applicable variant, without exact counts; unknown if inventory cannot confirm it. |
| `something similar but preservative-free?` | Search using LUMIFY context and the preference; offer only evidence-supported alternatives, or state the evidence is unavailable. |

Successful network behavior:

1. Browser POST `/apps/eye-agent` returns 200 `text/event-stream`, not HTML or a
   redirect. Shopify adds signed `shop`, `timestamp`, `path_prefix`, and `signature`
   query fields on the forwarded backend request; application queries stay intact.
2. Stream events are `id`, `chunk`, `message_complete`, `end_turn`. The ID is the
   existing signed bearer conversation token; do not paste it into public logs.
3. Later turns submit that same ID. Reload/history uses GET
   `/apps/eye-agent?history=true&conversation_id=...`, returning 200 JSON with messages.
4. Backend logs show `[chat] completed` with a Brain trace ID/status; Brain logs show
   `POST /v1/answer` 200. Audit entries keep evidence/citation IDs and the original answer.
5. The browser uses the storefront origin, requiring no cross-origin tunnel URL or
   CORS workaround. Proxy authentication does not depend on cookies (Shopify strips them).
6. SSE may arrive buffered through Shopify/tunnel infrastructure. Event order and
   completion must still be correct; this implementation sends the completed Brain
   answer as one chunk, not token-by-token model streaming. Actual intermediary
   buffering and timeout behavior remains part of the unlocked-storefront test.

References: [Shopify app proxy configuration](https://shopify.dev/docs/apps/build/online-store/app-proxies/index)
and [proxy authentication and forwarded parameters](https://shopify.dev/docs/apps/build/online-store/app-proxies/authenticate-app-proxies).

## Follow-up: false clarification after LUMIFY

The reported storefront conversation was present intact in SQLite. Replaying
`how much is that?` with that history reproduced the exact resolver defect:

```json
{"language":"en","references":["LUMIFY®"],"clarification":"What is the price of LUMIFY®?"}
```

The model had identified the product, but the application interpreted the rewritten
price question as a clarification request and skipped edsAI retrieval entirely.

Follow-up changes:

- `eyedropshop-brain/conversation.py`: replace the overloaded free-text field with
  explicit `standalone`, `resolved`, or `ambiguous` decisions; validate references
  against history; retry invalid decisions once; propagate model failures to the
  controlled unavailable response rather than falsely blaming ambiguous input.
- `eyedropshop-brain/model_provider.py`: support Ollama's JSON-schema output format
  for this resolver. Other generation calls retain their existing format.
- `eyedropshop-brain/tests/test_customer_upgrade.py`: cover resolved references plus
  a spurious clarification string, missing/hallucinated references, bare product
  names, and model outages. All 21 Python tests pass.

Restart the Brain using the normal demo restart to load these Python changes.
The existing conversation/session and edsAI-only factual boundary are retained.

Fresh-code live verification of both `how much is that?` and `how much is it?`
with the saved pre-failure history returned `answerable`, resolving LUMIFY and
retrieving its two variant prices from edsAI only. The already-running storefront
Brain process still requires a restart to load the fix.

## Follow-up: semantic retrieval-field planning

The reported `hoe much is CALMO® Eye Spray` turn was correctly understood by the
LLM as a CALMO price request, but the retrieval layer independently searched for
literal phrases such as `how much`, `price`, or `cost`. The misspelling therefore
caused it to omit the edsAI price and retrieve unrelated fields.

The query-understanding result now includes a schema-constrained list of required
edsAI fields (`price`, `inventory`, `warnings`, `usage`, `ingredients`, and so on).
Named-product retrieval uses that semantic plan rather than interpreting the raw
sentence a second time with keyword matching. The LLM chooses which facts it
needs; edsAI remains the only source of those facts.

Live verification confirmed that edsAI lists CALMO® Eye Spray at `39.95` without
a currency field. Fresh-code execution now answers the misspelled request with
that value and explicitly says the currency is unspecified.

A 22-case live language-understanding matrix covered price and availability typos,
informal price wording, combined price/stock requests, overview, warning, usage,
ingredients and preservative questions, itchy/red-eye discovery, eye pain, sudden
blurred vision, chemical exposure, product preferences, contact-lens wording, and
French price/availability requests. All meaning/safety expectations passed; one
initial test expectation was corrected because `anything for red eyes` is properly
a general product search rather than a personal-symptom statement.

Permanent regressions verify that semantic `price` selection reaches named-product
retrieval even when the raw request says `hoe much`, and that the deterministic
price response uses the retrieved evidence rather than rechecking exact wording.
The Python regression total is now 25; the React Router production build also passes.

## Follow-up: multi-product references and variant dates

In the red-eye conversation, the discovery answer offered LUMIFY and the Eye Allergy
Relief Kit. The singular follow-up `how much is it?` was resolved to both products.
The dialogue schema now records whether a reference is singular or plural. It
rejects a singular reference that expands to several products, and the resolver is
explicitly instructed that list order alone does not select a product. Live replay
now asks which product for `it` and resolves `both` to both products.

Live edsAI inspection confirmed that the two LUMIFY expiry dates belong to different
variants: 3.5 ml expires November 2027 and 7.5 ml expires February 2028. Expiry
evidence now carries the variant name in its text and metadata. Fresh-code live
execution reports each date with its corresponding size. The two age-related warning
statements shown in chat are both present in edsAI: a specific do-not-use statement
for children under 2 and a broader not-approved statement for people under 18.

The singular/plural reference and variant evidence changes raise the Python
regression total to 26 passing tests.

## Comprehensive product and language evaluation

Two repeatable evaluators now cover the catalog and natural-language layers
separately:

- `eyedropshop-brain/evaluate_customer_catalog.py` reads one consistent live
  edsAI snapshot and checks every variant's description, price, inventory,
  ingredients, warnings, usage, benefits/details, and expiry retrieval. It also
  checks every unique product title against the live matcher, verifies edsAI source
  metadata and non-empty evidence, rejects currency added to the numeric retail
  price, and requires variant labels on expiry evidence.
- `eyedropshop-brain/evaluate_customer_language.py` runs the production local LLM
  planner against direct, informal, misspelled, combined, and French questions.
  It covers every supported product-fact section plus discovery and safety routing.

Live results:

- 273 products, 272 unique titles, and 384 variants.
- 3,072 variant/section checks plus 272 title-matching checks: 3,344 total,
  zero failures.
- 48 live language cases: 38 product-fact cases and 10 discovery/safety cases,
  zero failures.
- 26 offline Python regressions pass, as do the Node route tests and production build.

The catalog snapshot contained price and inventory for all 384 variants. It contained
ingredients for 209, usage for 261, warnings for 91, benefits/details for 334, and
expiry/product-detail data for 159 variants. Missing fields were also verified: the
retrieval layer does not manufacture a passage when edsAI lacks that fact.

No finite evaluation can enumerate every possible human sentence or typo. This
suite instead exhausts the finite catalog/data paths and exercises representative
language families through the actual model. Reports are written to:

- `eyedropshop-brain/evaluation/customer_catalog_audit.json`
- `eyedropshop-brain/evaluation/customer_language_audit.json`

Repeat them with:

```sh
cd ~/Documents/EyeDropShop/Costumer_Agent/eyedropshop-brain
conda run --no-capture-output -n eyedropshop-brain python evaluate_customer_catalog.py
conda run --no-capture-output -n eyedropshop-brain python evaluate_customer_language.py
```

## Follow-up: new needs incorrectly treated as missing product references

The next reported turn, `my eyes are itchy, what drop you recommend?`, reproduced
as `resolution: ambiguous` despite being a self-contained new need. The resolver
was conflating searching for an unknown product with referring back to an
unidentified product.

`conversation.py` now explicitly distinguishes discovery/new topics from backward
references. Ambiguous decisions must include an unresolved expression copied from
the current question; standalone/resolved decisions must not. Medical safety and
product suitability remain downstream decisions, and the symptom text is preserved.
No symptom keyword router or external product knowledge was added.

`tests/test_customer_upgrade.py` now covers preserving a new symptom search without
attaching the previous product and rejecting ambiguity with no actual reference.
All 23 Python regressions pass. Fresh local-model replay resolves the reported
itchy-eye request as standalone with no clarification.

The broader live checks also preserved the LUMIFY price follow-up and resolved
`Could I use it for my itchy eyes?` to LUMIFY, while treating a new preservative-free
preference and shipping question as standalone. Full Brain execution classified
the reported turn as low-risk symptom discovery, searched edsAI, but the relevance
model selected zero candidates. The result was the existing professional-guidance
fallback, not the erroneous product clarification. This does not establish that no
suitable product exists in edsAI; it records this retrieval run's limitation.

`answer_generator.py` now also explains the missing supported match when this
fallback occurs. Its regression raises the Python total to 24 passing tests.
# Demo review fixes (September 27, 2026)

- Normalized shorthand citations such as `[1]` into the internal `[Source 1]` audit format before citation extraction. Both formats are removed from customer-visible responses in the route and theme renderer.
- Limited broad product overview answers to three complete sentences, including when edsAI supplies both description and benefits evidence.
- Strengthened high-risk symptom and chemical-exposure handling so the assistant tells the customer to seek urgent professional care now instead of offering a generic product disclaimer.
- Prevented duplicate submissions by disabling the chat input and send button until the active response completes.
- Added regressions for citation normalization, mixed-section overview length, urgent safety language, and numeric citation removal.
- Validation: 28 Python tests pass. Node tests and the production build could not be rerun in the final shell because `node` and `npm` were unavailable on its PATH; they previously passed before these focused JavaScript edits.
