/** Customer chat: all language generation and grounding are owned by the Brain. */
import { authenticate } from '../shopify.server';
import { saveMessage, getConversationHistory } from '../db.server';
import { createBrainService } from '../services/brain.server';
import { createSseStream } from '../services/streaming.server';
import { newConversationToken, validConversationToken } from '../services/conversation-token.server';

function headers(request) {
  const origin = request.headers.get('Origin');
  const allowed = (process.env.CHAT_ALLOWED_ORIGINS || '').split(',').map(s => s.trim()).filter(Boolean);
  const sameOrigin = origin === new URL(request.url).origin;
  if (origin && !sameOrigin && !allowed.includes(origin)) return null;
  return { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'Vary': 'Origin',
    ...(origin ? { 'Access-Control-Allow-Origin': origin } : {}),
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Accept, X-Shopify-Shop-Id' };
}
function json(data, status, h) { return new Response(JSON.stringify(data), {status, headers:h || {'Content-Type':'application/json'}}); }
function historyForBrain(messages) {
  return messages.filter(m => ['user', 'assistant'].includes(m.role)).slice(-8).map(m => {
    let content = m.content;
    try { const blocks = JSON.parse(content); if (Array.isArray(blocks)) content = blocks.filter(b => b.type === 'text').map(b => b.text).join('\n'); } catch { /* plain text */ }
    return {role:m.role, content:String(content || '').slice(0,6000)};
  }).filter(m => m.content);
}
export async function loader({request}) {
  await authenticate.public.appProxy(request);
  const h = headers(request);
  if (!h) return json({error:'Origin not allowed'},403);
  if (request.method === 'OPTIONS') return new Response(null,{status:204,headers:h});
  const url = new URL(request.url);
  const token = url.searchParams.get('conversation_id');
  if (!url.searchParams.has('history')) return json({error:'Use POST for chat'},405,h);
  if (!validConversationToken(token)) return json({error:'Invalid conversation'},403,h);
  return json({messages:await getConversationHistory(token)},200,h);
}
export async function action({request}) {
  await authenticate.public.appProxy(request);
  const h = headers(request);
  if (!h) return json({error:'Origin not allowed'},403);
  if (request.method === 'OPTIONS') return new Response(null,{status:204,headers:h});
  if (request.method !== 'POST') return json({error:'Use POST'},405,h);
  let body;
  try { body = await request.json(); } catch { return json({error:'Invalid JSON'},400,h); }
  if (!body || typeof body.message !== 'string' || !body.message.trim() || body.message.length > 2000)
    return json({error:'Enter a message of 1–2000 characters'},400,h);
  if (body.conversation_id && !validConversationToken(body.conversation_id))
    return json({error:'Invalid conversation. Start a new chat.'},403,h);
  const token = body.conversation_id || newConversationToken();
  const language = body.language === 'fr' ? 'fr' : 'en';
  const responseStream = createSseStream(async stream => {
    stream.sendMessage({type:'id',conversation_id:token});
    try {
      const history = historyForBrain(await getConversationHistory(token));
      await saveMessage(token,'user',body.message.trim());
      const result = await createBrainService().answer(body.message.trim(),language,history);
      // Brain has already validated and audited the original answer and citations.
      const answer = result.answer.replace(/\s*\[(?:Source\s+)?\d+\]/gi, '').trim();
      await saveMessage(token,'assistant',JSON.stringify([{type:'text',text:answer}]));
      console.info('[chat] completed', {trace_id:result.trace_id, status:result.status});
      stream.sendMessage({type:'chunk',chunk:answer});
      stream.sendMessage({type:'message_complete'});
    } catch {
      console.warn('[chat] dependency failure; check Brain and database health');
      stream.sendMessage({type:'chunk',chunk:language === 'fr'
        ? 'Le service est temporairement indisponible. Veuillez réessayer.'
        : 'The service is temporarily unavailable. Please try again.'});
    }
    stream.sendMessage({type:'end_turn'});
  });
  return new Response(responseStream,{headers:{...h,'Content-Type':'text/event-stream','X-Accel-Buffering':'no'}});
}
