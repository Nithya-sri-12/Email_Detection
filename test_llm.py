
import json
import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

MOCK_EMAILS = [
    {
        "uid": "001",
        "subject": "Q3 Budget Report - Urgent Review Needed",
        "sender": "alice.finance@company.com",
        "reply_to": "alice.finance@company.com",
        "body": """Hi,

I need you to review the Q3 budget report by EOD Friday. We're 15% over on the cloud infrastructure costs and the CFO wants an explanation and corrective action plan.

Please check sections 3.2 and 4.1 specifically. The board meeting is Monday morning.

Best,
Alice
Finance Team"""
    },
    {
        "uid": "002",
        "subject": "Your Amazon order has shipped!",
        "sender": "shipping-confirm@amazon.com",
        "reply_to": "shipping-confirm@amazon.com",
        "body": """Your order #112-4567890 has shipped!
Items: USB-C Hub x1, Mechanical Keyboard x1
Estimated delivery: Tomorrow by 8 PM
Track your package: amazon.com/track/112-4567890"""
    },
    {
        "uid": "003",
        "subject": "Quick question about the API integration",
        "sender": "bob.dev@startup.io",
        "reply_to": "bob.dev@startup.io",
        "body": """Hey,

We're integrating your REST API and running into a 401 error on the /v2/users endpoint even though we have a valid JWT. Is there a scope we're missing? 

We've tried reading the docs but the authentication section for v2 seems outdated.

Thanks,
Bob"""
    }
]

def parse_llm_json(raw: str, client) -> dict:
    """Parse model JSON robustly; ask model to repair malformed JSON if needed."""
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
            # Try common cleanup (e.g., trailing commas)
            cleaned = re.sub(r",\s*([}\]])", r"\1", cand)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

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
    repaired = client.generate_content(
        fix_prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=900),
    )
    repaired_text = (repaired.text or "").strip()
    if repaired_text.startswith("```"):
        repaired_text = re.sub(r"^```(?:json)?\s*", "", repaired_text, flags=re.IGNORECASE)
        repaired_text = re.sub(r"\s*```$", "", repaired_text)
    return json.loads(repaired_text)

def process_email(client, email_data):
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

    response = client.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(max_output_tokens=800),
    )
    raw = (response.text or "").strip()
    return parse_llm_json(raw, client)


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in .env")
        exit(1)

    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))

    print("=" * 60)
    print("AI EMAIL MANAGEMENT — LLM PIPELINE TEST")
    print("=" * 60)

    for i, email_data in enumerate(MOCK_EMAILS, 1):
        print(f"\n[Email {i}] Subject: {email_data['subject']}")
        print(f"          From:    {email_data['sender']}")
        print("          Processing with Gemini...")

        result = process_email(client, email_data)

        priority_sym = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(result['priority'], "🟡")
        cat_sym = {"Work":"💼","Personal":"👤","Spam":"🚫","Finance":"💰",
                   "Support":"🛠️","Newsletter":"📰","Other":"📧"}.get(result['category'],"📧")

        print(f"\n  {priority_sym} Priority : {result['priority']}")
        print(f"  {cat_sym} Category : {result['category']}")
        print(f"\n  📝 Summary:\n     {result['summary']}")
        print(f"\n  ✉️  Draft Reply:\n     {result['draft_reply']}")
        print("\n" + "-" * 60)

    print("\n✅ LLM pipeline test complete! All 3 emails processed.")
    print("   Next: run bot.py with your Gmail + Discord credentials.")



