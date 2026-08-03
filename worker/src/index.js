/**
 * =============================================================================
 * telegram-anima-diffusion-bot :: Cloudflare Worker :: webhook router
 * =============================================================================
 * Pure-JS, no Node server, runs on Cloudflare Workers edge runtime.
 *
 * Flow:
 *   1. Telegram webhook -> this worker
 *   2. Verify user.id against ALLOWED_TELEGRAM_USER_ID
 *   3. Parse /generate <prompt>
 *   4. Send inline-keyboard confirmation to user
 *   5. Fire repository_dispatch to GitHub Actions
 *   6. Return 200 instantly (avoid Telegram 5s retry)
 * =============================================================================
 */

const TELEGRAM_API = (token) => `https://api.telegram.org/bot${token}`;

// ---------------------------------------------------------------------------
//  Main fetch handler (Worker entrypoint)
// ---------------------------------------------------------------------------
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === "/" || url.pathname === "/health") {
      return json({ ok: true, service: "telegram-anima-diffusion-bot", time: Date.now() });
    }

    // Telegram callback (button presses)
    if (url.pathname === "/callback" && request.method === "POST") {
      return handleCallback(request, env, ctx);
    }

    // Telegram webhook
    if (url.pathname === "/webhook" && request.method === "POST") {
      return handleWebhook(request, env, ctx);
    }

    return json({ error: "not found" }, 404);
  }
};

// ---------------------------------------------------------------------------
//  Webhook handler
// ---------------------------------------------------------------------------
async function handleWebhook(request, env, ctx) {
  let body;
  try {
    body = await request.json();
  } catch (e) {
    return json({ ok: true }); // swallow malformed payloads
  }

  // Empty body / polling artifact
  if (!body || (!body.message && !body.edited_message && !body.callback_query)) {
    return json({ ok: true });
  }

  // Handle callback queries (button presses)
  if (body.callback_query) {
    return handleCallbackQuery(body.callback_query, env);
  }

  const msg = body.message || body.edited_message;
  if (!msg || !msg.text) return json({ ok: true });

  const fromId = String(msg.from?.id || "");
  const allowedId = String(env.ALLOWED_TELEGRAM_USER_ID || "");

  // ---- STRICT AUTH -------------------------------------------------------
  if (!fromId || fromId !== allowedId) {
    // Silent 200 to drop the request (don't reveal we exist)
    return json({ ok: true });
  }

  const chatId = msg.chat.id;
  const text = msg.text.trim();

  // /start
  if (text === "/start") {
    await tg(env, "sendMessage", {
      chat_id: chatId,
      text: "👋 *Anima T2I Bot*\n\nSend me:\n`/generate <your prompt>`\n\nExamples:\n`/generate 1girl, anime, vibrant`\n`/generate cyberpunk city at night, neon`",
      parse_mode: "Markdown"
    });
    return json({ ok: true });
  }

  // /help
  if (text === "/help") {
    await tg(env, "sendMessage", {
      chat_id: chatId,
      text: "*Commands*\n/generate <prompt> — generate image\n/cancel — cancel latest run\n/status — show runner status",
      parse_mode: "Markdown",
      reply_markup: inlineKeyboard([
        [{ text: "🔍 Check Status", callback_data: "status" }, { text: "❌ Cancel", callback_data: "cancel" }]
      ])
    });
    return json({ ok: true });
  }

  // /generate <prompt>
  if (text.startsWith("/generate")) {
    const prompt = text.slice("/generate".length).trim();
    if (!prompt) {
      await tg(env, "sendMessage", {
        chat_id: chatId,
        text: "❌ Usage: `/generate <prompt>`",
        parse_mode: "Markdown"
      });
      return json({ ok: true });
    }
    return await dispatchGeneration(env, ctx, chatId, prompt);
  }

  // Free-text (treat any non-command message as a prompt)
  if (!text.startsWith("/")) {
    return await dispatchGeneration(env, ctx, chatId, text);
  }

  // Unknown command
  await tg(env, "sendMessage", {
    chat_id: chatId,
    text: "Unknown command. Try /help",
    parse_mode: "Markdown"
  });
  return json({ ok: true });
}

// ---------------------------------------------------------------------------
//  Dispatch generation to GitHub Actions
// ---------------------------------------------------------------------------
async function dispatchGeneration(env, ctx, chatId, prompt) {
  // 1. Send instant confirmation with inline buttons
  await tg(env, "sendMessage", {
    chat_id: chatId,
    text:
      `🎨 *Queued!*\n\n` +
      `*Prompt:* \`${truncate(prompt, 200)}\`\n\n` +
      `⏳ Spinning up GitHub Actions runner...\n` +
      `⏱ ETA: ~10-15 min on CPU`,
    parse_mode: "Markdown",
    reply_markup: inlineKeyboard([
      [
        { text: "🔍 Check Status", callback_data: "status" },
        { text: "❌ Cancel Run",   callback_data: "cancel" }
      ]
    ])
  });

  // 2. Fire repository_dispatch (don't block the response on this)
  ctx.waitUntil(
    (async () => {
      try {
        const r = await fetch(
          `https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`,
          {
            method: "POST",
            headers: {
              "Accept": "application/vnd.github+json",
              "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
              "X-GitHub-Api-Version": "2022-11-28",
              "User-Agent": "telegram-anima-bot"
            },
            body: JSON.stringify({
              event_type: "trigger_anima_generation",
              client_payload: {
                prompt: prompt,
                chat_id: String(chatId),
                ts: String(Date.now())
              }
            })
          }
        );
        if (!r.ok) {
          const errText = await r.text();
          await tg(env, "sendMessage", {
            chat_id: chatId,
            text: `⚠️ GitHub dispatch failed: ${r.status}\n\`${truncate(errText, 200)}\``,
            parse_mode: "Markdown"
          });
        }
      } catch (e) {
        await tg(env, "sendMessage", {
          chat_id: chatId,
          text: `⚠️ Dispatch error: \`${String(e).slice(0, 200)}\``,
          parse_mode: "Markdown"
        });
      }
    })()
  );

  // 3. Return 200 immediately so Telegram doesn't retry
  return json({ ok: true });
}

// ---------------------------------------------------------------------------
//  Callback query handler (button presses)
// ---------------------------------------------------------------------------
async function handleCallbackQuery(cbq, env) {
  const data = cbq.data || "";
  const chatId = cbq.message?.chat?.id;
  const cbId   = cbq.id;

  // Always answer the callback to remove the loading spinner
  const answer = (text) =>
    fetch(`${TELEGRAM_API(env.TELEGRAM_BOT_TOKEN)}/answerCallbackQuery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ callback_query_id: cbId, text, show_alert: false })
    });

  if (data === "status") {
    await answer("Checking latest run...");
    const runs = await getLatestRun(env);
    if (!runs) {
      await tg(env, "sendMessage", {
        chat_id: chatId,
        text: "No recent runs found.",
        parse_mode: "Markdown"
      });
    } else {
      const status = runs.status || "unknown";
      const emoji = status === "completed" ? "✅" : status === "in_progress" ? "🔄" : status === "queued" ? "⏳" : "❌";
      const html_url = runs.html_url || "N/A";
      await tg(env, "sendMessage", {
        chat_id: chatId,
        text:
          `${emoji} *Run status:* \`${status}\`\n` +
          `*Conclusion:* \`${runs.conclusion || "—"}\`\n` +
          `*Run ID:* \`${runs.id}\`\n\n` +
          `[View on GitHub](${html_url})`,
        parse_mode: "Markdown",
        disable_web_page_preview: true
      });
    }
  } else if (data === "cancel") {
    await answer("Cancelling latest run...");
    const runs = await getLatestRun(env, ["in_progress", "queued"]);
    if (!runs) {
      await tg(env, "sendMessage", {
        chat_id: chatId,
        text: "No active run to cancel.",
        parse_mode: "Markdown"
      });
    } else {
      const r = await fetch(
        `https://api.github.com/repos/${env.GITHUB_REPO}/actions/runs/${runs.id}/cancel`,
        {
          method: "POST",
          headers: {
            "Accept": "application/vnd.github+json",
            "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "telegram-anima-bot"
          }
        }
      );
      if (r.ok) {
        await tg(env, "sendMessage", {
          chat_id: chatId,
          text: `✅ Cancelled run #${runs.id}`,
          parse_mode: "Markdown"
        });
      } else {
        await tg(env, "sendMessage", {
          chat_id: chatId,
          text: `⚠️ Cancel failed: ${r.status}`,
          parse_mode: "Markdown"
        });
      }
    }
  } else {
    await answer("Unknown action");
  }

  return json({ ok: true });
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
async function tg(env, method, payload) {
  return fetch(`${TELEGRAM_API(env.TELEGRAM_BOT_TOKEN)}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

async function getLatestRun(env, statuses = null) {
  const url = new URL(`https://api.github.com/repos/${env.GITHUB_REPO}/actions/runs`);
  url.searchParams.set("per_page", "10");
  if (statuses && statuses.length === 1) {
    url.searchParams.set("status", statuses[0]);
  }
  const r = await fetch(url, {
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": `Bearer ${env.GITHUB_TOKEN}`,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "telegram-anima-bot"
    }
  });
  if (!r.ok) return null;
  const data = await r.json();
  const runs = data.workflow_runs || [];
  if (runs.length === 0) return null;
  if (!statuses) return runs[0];
  return runs.find(x => statuses.includes(x.status)) || null;
}

function inlineKeyboard(rows) {
  return { inline_keyboard: rows };
}

function truncate(s, n) {
  s = String(s || "");
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json" }
  });
}
