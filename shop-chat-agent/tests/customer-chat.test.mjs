import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {newConversationToken, validConversationToken} from '../app/services/conversation-token.server.js';
import {createBrainService} from '../app/services/brain.server.js';
import {createSseStream} from '../app/services/streaming.server.js';
process.env.CHAT_SESSION_SECRET='offline-test-only-'.repeat(3);
process.env.BRAIN_CUSTOMER_API_KEY='test-placeholder';
process.env.CHAT_ALLOWED_ORIGINS='https://store.example';
let saved=[], received;
globalThis.chatTest={authenticate:{public:{appProxy:async request=>{if(request.headers.get('X-Test-Invalid-Proxy')) throw new Response(null,{status:401});}}},saveMessage:async(...a)=>saved.push(a), getConversationHistory:async()=>[{role:'user',content:'Compare Alpha and Beta'}],
  createBrainService:()=>({answer:async(...args)=>{if(globalThis.failBrain) throw new Error('private connection details'); received=args;return {answer:'Grounded comparison [1] [Source 1]',trace_id:'test-trace',status:'answerable',evidence:[{source_title:'Alpha',metadata:{product_title:'Alpha',product_id:'22',variant_id:'gid://shopify/ProductVariant/123',variant_title:'10 ml'}}]};}}),
  createSseStream,newConversationToken,validConversationToken};
let source=await readFile(new URL('../app/routes/chat.jsx',import.meta.url),'utf8');
source=source.replace(/^import .*;$/gm,'');
source='const {authenticate,saveMessage,getConversationHistory,createBrainService,createSseStream,newConversationToken,validConversationToken}=globalThis.chatTest;\n'+source;
const route=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
function request(body,origin='https://store.example') { return new Request('https://chat.example/chat',{method:'POST',headers:{Origin:origin,'Content-Type':'application/json'},body:JSON.stringify(body)}); }
test('signed tokens resist guessing and tampering',()=>{
 const token=newConversationToken(); assert.ok(validConversationToken(token));
 assert.equal(validConversationToken('1234'),false);
 assert.equal(validConversationToken(token.slice(0,-1)+(token.endsWith('a')?'b':'a')),false);
});
test('unknown origin is rejected',async()=>assert.equal((await route.action({request:request({message:'hi'},'https://other.example')})).status,403));
test('invalid history token is rejected',async()=>{
 const r=await route.loader({request:new Request('https://chat.example/chat?history=true&conversation_id=123')}); assert.equal(r.status,403);
});
test('conversation history and French reach Brain; stream completes',async()=>{
 const r=await route.action({request:request({message:'Lequel est moins cher ?',language:'fr'})});
 const body=await r.text(); assert.ok(body.includes('Grounded comparison')); assert.ok(body.includes('end_turn'));
 assert.ok(!body.includes('[1]'));
 assert.ok(body.includes('product_results')); assert.ok(body.includes('"variant_id":"123"'));
 assert.equal(received[1],'fr'); assert.equal(received[2][0].content,'Compare Alpha and Beta'); assert.equal(saved.length,2);
});
test('oversized input is rejected',async()=>assert.equal((await route.action({request:request({message:'x'.repeat(2001)})})).status,400));
test('Brain service sends bounded history contract',async()=>{
 const old=globalThis.fetch;
 globalThis.fetch=async(url,options)=>{ const body=JSON.parse(options.body); assert.equal(body.history[0].content,'Alpha'); assert.equal(body.language,'fr'); assert.ok(options.signal);return {ok:true,json:async()=>({answer:'ok'})}; };
 try {assert.equal((await createBrainService().answer('Prix?','fr',[{role:'user',content:'Alpha'}])).answer,'ok');} finally {globalThis.fetch=old;}
});

test('dependency failures return a controlled message without private errors',async()=>{
 globalThis.failBrain=true;
 try {const r=await route.action({request:request({message:'Price of Alpha?'})});const text=await r.text();assert.ok(text.includes('temporarily unavailable'));assert.ok(text.includes('end_turn'));assert.ok(!text.includes('private connection'));}
 finally {globalThis.failBrain=false;}
});

test('proxy authentication rejects requests before chat processing',async()=>{
 const req=request({message:'hello'});req.headers.set('X-Test-Invalid-Proxy','1');
 await assert.rejects(route.action({request:req}),error=>error.status===401);
});
test('customer display strips markers after Brain processing',async()=>{
 const response=await route.action({request:request({message:'What is Alpha?'})});
 assert.ok(!(await response.text()).includes('[Source 1]'));
 assert.ok(!saved.at(-1)[2].includes('[Source 1]'));
 assert.ok(!saved.at(-1)[2].includes('[1]'));
});
test('theme chat and history use the proxy root, with query preserved',async()=>{
 const js=await readFile(new URL('../extensions/chat-bubble/assets/chat.js',import.meta.url),'utf8');
 const fn=js.slice(js.indexOf('  function apiUrl('),js.indexOf('  // Old demo'));
 const url=new Function('window',fn+';return apiUrl;')({shopChatConfig:{proxyBaseUrl:'/apps/eye-agent'}});
 assert.equal(url(''),'/apps/eye-agent');
 assert.equal(url('?history=true&conversation_id=abc'),'/apps/eye-agent?history=true&conversation_id=abc');
});
test('theme adds variants through the Shopify Ajax cart API',async()=>{
 const js=await readFile(new URL('../extensions/chat-bubble/assets/chat.js',import.meta.url),'utf8');
 assert.ok(js.includes('cart/add.js'));
 assert.ok(js.includes('items: [{id: Number(variantId), quantity: 1}]'));
 assert.ok(!js.includes('input.value = `Add ${product.title} to my cart`'));
});
