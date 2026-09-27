import { createHmac, randomUUID, timingSafeEqual } from 'node:crypto';
function signature(id) {
  const key = process.env.CHAT_SESSION_SECRET;
  if (!key || key.length < 32) throw new Error('Configure CHAT_SESSION_SECRET (at least 32 characters)');
  return createHmac('sha256', key).update(id).digest('hex');
}
export function newConversationToken() {
  const id = randomUUID();
  return `${id}.${signature(id)}`;
}
export function validConversationToken(token) {
  if (typeof token !== 'string' || !/^[a-f0-9-]{36}\.[a-f0-9]{64}$/.test(token)) return false;
  const [id, mac] = token.split('.');
  return timingSafeEqual(Buffer.from(mac, 'hex'), Buffer.from(signature(id), 'hex'));
}
