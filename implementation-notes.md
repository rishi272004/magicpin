# Vera message engine - implementation notes

## What the task is
Build a deterministic message engine for magicpin's Vera assistant. The engine must:
- Store pushed context (category, merchant, customer, trigger).
- Decide when to send a message.
- Compose a message with the right tone, specificity, and CTA.
- Respond to merchant or customer replies.

## What I built
I implemented a small FastAPI server in bot.py that exposes the required endpoints:
- GET /v1/healthz
- GET /v1/metadata
- POST /v1/context
- POST /v1/tick
- POST /v1/reply

The core message engine is a deterministic compose() function with rule-based templates per trigger kind.

## How it works (simple steps)
1. Context storage
   - POST /v1/context stores the latest version for each scope and id.
2. Trigger selection
   - POST /v1/tick receives active trigger ids.
   - The bot picks one best trigger per merchant or customer using a priority score.
3. Message composition
   - The compose() function uses category, merchant, trigger, and customer data.
   - Each trigger kind maps to a specific template and a single CTA.
4. Reply handling
   - POST /v1/reply detects stop requests, auto-replies, and positive intent.
   - It sends a short follow-up or ends the conversation.

## Why this meets the rubric
- Decision quality: highest-priority trigger is chosen per target.
- Specificity: messages include concrete numbers and facts from the trigger payload.
- Category fit: templates avoid promotional claims and use domain-appropriate wording.
- Merchant fit: active offers and merchant metrics are reused when available.
- Engagement compulsion: each message ends with one clear next action.

## How to run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Start the server:
   uvicorn bot:app --host 0.0.0.0 --port 8080
3. Run the judge simulator:
   python judge_simulator.py
