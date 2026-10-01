
import asyncio
import imaplib
import smtplib
import email
import os
import json
import time
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import decode_header
from typing import Optional
import discord
from discord.ext import tasks
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GMAIL_USER     = (os.getenv("GMAIL_USER") or "").strip().strip('"').strip("'")

GMAIL_PASS     = (os.getenv("GMAIL_APP_PASSWORD") or "").strip().strip('"').strip("'").replace(" ", "")
DISCORD_TOKEN  = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
DISCORD_CHANNEL= int(os.getenv("DISCORD_CHANNEL_ID", "0"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
IMAP_TIMEOUT   = int(os.getenv("IMAP_TIMEOUT_SECONDS", "20"))


pending_approvals: dict[int, dict] = {}
email_context: list[dict] = []
seen_uids: set[str] = set()
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)

def parse_llm_json(raw: str) -> Optional[dict]:
    """Parse model JSON robustly; returns None if unrecoverable."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(text[first:last + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    try:
        fix_prompt = f"""Fix this invalid JSON so it becomes valid JSON.
Return ONLY valid JSON with this exact schema:
{{
  "category": "<one of: Work, Personal, Spam, Finance, Support, Newsletter, Other>",
  "summary": "<2-3 sentence concise summary>",
  "draft_reply": "<professional, helpful draft reply>",
  "priority": "<High | Medium | Low>"
}}

Invalid JSON:
{text}
"""
        repaired = gemini_model.generate_content(
            fix_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=900),
        )
        repaired_text = (repaired.text or "").strip()
        if repaired_text.startswith("```"):
            repaired_text = re.sub(r"^```(?:json)?\s*", "", repaired_text, flags=re.IGNORECASE)
            repaired_text = re.sub(r"\s*```$", "", repaired_text)
        return json.loads(repaired_text)
    except Exception:
        return None


def decode_str(s: str) -> str:
    """Decode encoded email header string."""
    parts = decode_header(s)
    result = []
    for part, enc in parts:
        if isinstance(part, bytes):
            result.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            result.append(part)
    return "".join(result)


def fetch_new_emails() -> list[dict]:
    """Connect to Gmail via IMAP and fetch unseen emails."""
    emails = []
    try:
        print("[IMAP] Connecting to imap.gmail.com...")
        mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=IMAP_TIMEOUT)
        print("[IMAP] Logging in...")
        mail.login(GMAIL_USER, GMAIL_PASS)
        print("[IMAP] Selecting inbox...")
        status, _ = mail.select("inbox")
        if status != "OK":
            print(f"[IMAP] Failed to select inbox (status={status})")
            mail.logout()
            return emails

        print("[IMAP] Searching for UNSEEN emails...")
        status, data = mail.search(None, "UNSEEN")
        if status != "OK":
            print(f"[IMAP] Search failed (status={status})")
            mail.logout()
            return emails
        uid_list = data[0].split()
        print(f"[IMAP] Found {len(uid_list)} unseen email(s).")

        for uid in uid_list:
            uid_str = uid.decode()
            if uid_str in seen_uids:
                continue

            _, msg_data = mail.fetch(uid, "(RFC822)")
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            subject = decode_str(msg.get("Subject", "(no subject)"))
            sender  = decode_str(msg.get("From", "unknown"))
            reply_to = msg.get("Reply-To") or sender

            # Extract plain-text body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ct = part.get_content_type()
                    cd = str(part.get("Content-Disposition", ""))
                    if ct == "text/plain" and "attachment" not in cd:
                        charset = part.get_content_charset() or "utf-8"
                        body = part.get_payload(decode=True).decode(charset, errors="replace")
                        break
            else:
                charset = msg.get_content_charset() or "utf-8"
                body = msg.get_payload(decode=True).decode(charset, errors="replace")

            emails.append({
                "uid":      uid_str,
                "subject":  subject,
                "sender":   sender,
                "reply_to": reply_to,
                "body":     body[:3000],  
            })
            seen_uids.add(uid_str)

        print("[IMAP] Logout.")
        mail.logout()
    except Exception as e:
        print(f"[IMAP ERROR] {e}")
    return emails


def send_reply(to_address: str, subject: str, body: str) -> bool:
    """Send reply via Gmail SMTP."""
    try:
        msg = MIMEMultipart()
        msg["From"]    = GMAIL_USER
        msg["To"]      = to_address
        msg["Subject"] = f"Re: {subject}" if not subject.startswith("Re:") else subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_USER, GMAIL_PASS)
            server.sendmail(GMAIL_USER, to_address, msg.as_string())
        return True
    except Exception as e:
        print(f"[SMTP ERROR] {e}")
        return False


# LLM PROCESSING

def process_email_with_llm(email_data: dict) -> dict:
    """
    Call Gemini to:
      1. Classify the email
      2. Summarize it
      3. Draft a reply
    Returns enriched dict with category, summary, draft_reply.
    """
    prompt = f"""You are an intelligent email assistant. Analyze the following email and respond ONLY with valid JSON.

Email Details:
- From: {email_data['sender']}
- Subject: {email_data['subject']}
- Body:
{email_data['body']}

Respond with this exact JSON structure:
{{
  "category": "<one of: Work, Personal, Spam, Finance, Support, Newsletter, Other>",
  "summary": "<2-3 sentence concise summary of what this email is about>",
  "draft_reply": "<a professional, helpful draft reply to this email (3-6 sentences)>",
  "priority": "<High | Medium | Low>"
}}

Return ONLY the JSON, no markdown fences, no extra text."""

    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=800),
    )
    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    result = parse_llm_json(raw)
    if result is None:
        # Fallback if JSON is malformed
        result = {
            "category":    "Other",
            "summary":     "Could not parse LLM response.",
            "draft_reply": "Thank you for your email. I will get back to you shortly.",
            "priority":    "Medium"
        }

    email_data.update(result)
    return email_data


def answer_question_with_llm(question: str) -> str:
    """Answer a user Q&A question about stored email context."""
    if not email_context:
        return "No emails have been processed yet."

    context_str = "\n\n---\n\n".join([
        f"Email #{i+1}\n"
        f"From: {e['sender']}\n"
        f"Subject: {e['subject']}\n"
        f"Category: {e.get('category','?')}\n"
        f"Summary: {e.get('summary','?')}\n"
        f"Body snippet: {e['body'][:400]}"
        for i, e in enumerate(email_context[-20:])  # last 20 emails
    ])

    prompt = f"""You are an email assistant with access to the user's recent inbox.

Here are the processed emails:
{context_str}

User question: {question}

Answer concisely and helpfully based on the emails above. 
If the answer isn't in the emails, say so clearly."""

    response = gemini_model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=500),
    )
    return (response.text or "").strip()


# DISCORD BOT 

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)


async def resolve_target_channel() -> Optional[discord.abc.Messageable]:
    """Resolve configured Discord channel via cache, then API fallback."""
    channel = bot.get_channel(DISCORD_CHANNEL)
    if channel is not None:
        return channel
    try:
        fetched = await bot.fetch_channel(DISCORD_CHANNEL)
        return fetched
    except Exception as e:
        print(f"[DISCORD] Could not fetch channel {DISCORD_CHANNEL}: {e}")
        return None


def format_discord_message(e: dict) -> str:
    """Format processed email into a Discord embed-style message."""
    priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(e.get("priority","Medium"), "🟡")
    category_emoji = {
        "Work": "💼", "Personal": "👤", "Spam": "🚫", "Finance": "💰",
        "Support": "🛠️", "Newsletter": "📰", "Other": "📧"
    }.get(e.get("category","Other"), "📧")

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📬 **NEW EMAIL**  {priority_emoji} `{e.get('priority','Medium')} Priority`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**From:** `{e['sender']}`\n"
        f"**Subject:** {e['subject']}\n\n"
        f"{category_emoji} **Category:** `{e.get('category','?')}`\n\n"
        f"📝 **Summary:**\n{e.get('summary','')}\n\n"
        f"✉️ **Draft Reply:**\n```\n{e.get('draft_reply','')}\n```\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"React with ✅ to **send** this reply  |  ❌ to **ignore**\n"
        f"Or type `yes` / `no` in reply to this message."
    )


async def post_email_to_discord(channel, email_data: dict):
    """Post a processed email card to Discord and set up approval tracking."""
    msg_text = format_discord_message(email_data)
    sent_msg = await channel.send(msg_text)

    # Add reaction shortcuts
    await sent_msg.add_reaction("✅")
    await sent_msg.add_reaction("❌")

    pending_approvals[sent_msg.id] = email_data
    print(f"[DISCORD] Posted email: {email_data['subject'][:50]}")


@bot.event
async def on_ready():
    print(f"[BOT] Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"[BOT] Watching channel ID: {DISCORD_CHANNEL}")
    guild_info = ", ".join([f"{g.name}({g.id})" for g in bot.guilds]) or "none"
    print(f"[BOT] Connected guilds: {guild_info}")
    email_poller.start()


@bot.event
async def on_message(message: discord.Message):
    """Handle text-based yes/no approvals and Q&A questions."""
    if message.author.bot:
        return
    if message.channel.id != DISCORD_CHANNEL:
        return

    content = message.content.strip().lower()

    # Q&A: ask <question> 
    if content.startswith("?ask "):
        question = message.content[5:].strip()
        await message.channel.send(f"🔍 Searching email context for: *{question}*...")
        answer = await asyncio.to_thread(answer_question_with_llm, question)
        await message.channel.send(f"🤖 **Answer:**\n{answer}")
        return

    # list: show pending approvals 
    if content == "?list":
        if not pending_approvals:
            await message.channel.send("✅ No pending email approvals.")
        else:
            lines = [f"`#{mid}` — **{d['subject'][:60]}** (from {d['sender'][:40]})"
                     for mid, d in pending_approvals.items()]
            await message.channel.send("**Pending approvals:**\n" + "\n".join(lines))
        return

    # help 
    if content == "?help":
        await message.channel.send(
            "**📬 Email AI Bot Commands**\n"
            "━━━━━━━━━━━━━━━━━━━━━━━\n"
            "React ✅ on an email card → send the draft reply\n"
            "React ❌ on an email card → ignore\n"
            "`yes` / `no` (as a reply to an email card) → same as above\n"
            "`?ask <question>` → Q&A about your inbox\n"
            "`?list` → show pending approvals\n"
            "`?help` → show this message"
        )
        return

    #  yes/no as a text reply to an email message ──
    if message.reference and content in ("yes", "no"):
        ref_id = message.reference.message_id
        if ref_id not in pending_approvals:
            await message.channel.send("⚠️ Could not find the email for that message.")
            return
        email_data = pending_approvals.pop(ref_id)
        await _handle_approval(message.channel, email_data, approved=(content == "yes"))


@bot.event
async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
    """Handle ✅/❌ reactions on email cards."""
    if user.bot:
        return
    if reaction.message.channel.id != DISCORD_CHANNEL:
        return

    msg_id = reaction.message.id
    if msg_id not in pending_approvals:
        return

    emoji = str(reaction.emoji)
    if emoji not in ("✅", "❌"):
        return

    email_data = pending_approvals.pop(msg_id)
    approved = (emoji == "✅")
    await _handle_approval(reaction.message.channel, email_data, approved)


async def _handle_approval(channel, email_data: dict, approved: bool):
    """Execute the approval decision."""
    if approved:
        success = await asyncio.to_thread(
            send_reply,
            email_data["reply_to"],
            email_data["subject"],
            email_data["draft_reply"]
        )
        if success:
            await channel.send(
                f"✅ **Reply sent** to `{email_data['reply_to']}`\n"
                f"Subject: *{email_data['subject']}*"
            )
        else:
            await channel.send(
                f"❌ **Failed to send** reply to `{email_data['reply_to']}`. Check SMTP config."
            )
    else:
        await channel.send(
            f"🚫 **Ignored** email from `{email_data['sender']}`\n"
            f"Subject: *{email_data['subject']}*"
        )


# EMAIL POLLING LOOP 

@tasks.loop(seconds=POLL_INTERVAL)
async def email_poller():
    """Background task: poll Gmail, process with LLM, post to Discord."""
    channel = await resolve_target_channel()
    if channel is None:
        print(f"[POLLER] Channel {DISCORD_CHANNEL} not found. Check channel ID and bot access.")
        return

    print(f"[POLLER] Checking inbox...")
    new_emails = await asyncio.to_thread(fetch_new_emails)

    for raw_email in new_emails:
        print(f"[POLLER] Processing: {raw_email['subject'][:60]}")
        processed = await asyncio.to_thread(process_email_with_llm, raw_email)
        email_context.append(processed)
        await post_email_to_discord(channel, processed)

    if new_emails:
        print(f"[POLLER] Processed {len(new_emails)} new email(s).")
    else:
        print("[POLLER] No new unseen emails.")


@email_poller.before_loop
async def before_poller():
    await bot.wait_until_ready()


# ENTRYPOINT 

if __name__ == "__main__":
    required = ["GMAIL_USER", "GMAIL_APP_PASSWORD", "DISCORD_BOT_TOKEN",
                "DISCORD_CHANNEL_ID", "GEMINI_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("        Copy .env.example to .env and fill in values.")
        exit(1)

    print("[BOT] Starting AI Email Management Bot...")
    bot.run(DISCORD_TOKEN)



