import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Response
from pydantic import BaseModel

app = FastAPI()
START_TIME = time.time()

# In-memory stores
contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
conversations: Dict[str, Dict[str, Any]] = {}
sent_suppression_keys: set[str] = set()
auto_reply_tracker: Dict[str, Dict[str, Any]] = {}


class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str]


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.get("/v1/healthz")
async def healthz() -> Dict[str, Any]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        counts[scope] = counts.get(scope, 0) + 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.head("/v1/healthz")
async def healthz_head() -> Response:
    return Response(status_code=200)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.head("/")
async def root_head() -> Response:
    return Response(status_code=200)


@app.get("/v1/metadata")
async def metadata() -> Dict[str, Any]:
    return {
        "team_name": "Vera Challenge Bot",
        "team_members": ["Candidate"],
        "model": "rules-deterministic",
        "approach": "deterministic rule-based composer with trigger-aware templates",
        "contact_email": "candidate@example.com",
        "version": "1.0.0",
        "submitted_at": "2026-04-30T00:00:00Z",
    }


@app.post("/v1/context")
async def push_context(body: ContextBody) -> Dict[str, Any]:
    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": current["version"],
        }

    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.scope}_{body.context_id}_{body.version}",
        "stored_at": body.delivered_at,
    }


@app.post("/v1/tick")
async def tick(body: TickBody) -> Dict[str, Any]:
    actions: List[Dict[str, Any]] = []
    now = _parse_dt(body.now)

    candidates: List[Tuple[int, Dict[str, Any]]] = []
    for trigger_id in body.available_triggers:
        trigger = _get_context("trigger", trigger_id)
        if not trigger:
            continue
        if _is_expired(trigger, now):
            continue
        suppression_key = trigger.get("suppression_key")
        if suppression_key and suppression_key in sent_suppression_keys:
            continue

        score = _trigger_priority_score(trigger)
        candidates.append((score, trigger))

    # One message per target (merchant or customer) per tick.
    grouped: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for score, trigger in candidates:
        target_key = _target_key(trigger)
        current = grouped.get(target_key)
        if current is None or score > current[0]:
            grouped[target_key] = (score, trigger)

    # Sort by priority score descending for deterministic order.
    sorted_triggers = sorted(
        grouped.values(), key=lambda item: (-item[0], item[1].get("id", ""))
    )

    for _, trigger in sorted_triggers:
        merchant = _get_context("merchant", trigger.get("merchant_id", ""))
        if not merchant:
            continue
        category = _get_context("category", merchant.get("category_slug", ""))
        if not category:
            continue
        customer = None
        if trigger.get("scope") == "customer":
            customer_id = trigger.get("customer_id")
            customer = _get_context("customer", customer_id or "")
            if not customer:
                continue

        composed = compose(category, merchant, trigger, customer)
        conversation_id = f"conv_{trigger.get('id', '')}"
        conversations[conversation_id] = {
            "trigger_id": trigger.get("id"),
            "trigger_kind": trigger.get("kind"),
            "merchant_id": trigger.get("merchant_id"),
            "customer_id": trigger.get("customer_id"),
            "auto_reply_hits": 0,
            "turns": [],
        }

        template_name = f"vera_{trigger.get('kind', 'generic')}_v1"
        template_params = _template_params(merchant, trigger)
        action = {
            "conversation_id": conversation_id,
            "merchant_id": trigger.get("merchant_id"),
            "customer_id": trigger.get("customer_id"),
            "send_as": composed["send_as"],
            "trigger_id": trigger.get("id"),
            "template_name": template_name,
            "template_params": template_params,
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": composed["suppression_key"],
            "rationale": composed["rationale"],
        }
        actions.append(action)
        if composed.get("suppression_key"):
            sent_suppression_keys.add(composed["suppression_key"])

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody) -> Dict[str, Any]:
    conv = conversations.get(body.conversation_id, {})
    conv.setdefault("turns", []).append({"from": body.from_role, "body": body.message})
    conversations[body.conversation_id] = conv

    trigger = _get_context("trigger", conv.get("trigger_id", "")) if conv else None
    trigger_kind = conv.get("trigger_kind") or (trigger.get("kind") if trigger else None)
    merchant = _get_context("merchant", body.merchant_id) or {}
    category = _get_context("category", merchant.get("category_slug", "")) or {}
    customer = _get_context("customer", body.customer_id) if body.customer_id else None

    normalized = _normalize_text(body.message)

    if _is_stop_message(normalized):
        return {"action": "end", "rationale": "User asked to stop."}

    if _is_auto_reply(normalized, conv):
        hits = _record_auto_reply(body.merchant_id, normalized)
        if _is_repeat_in_conversation(conv, normalized) or hits >= 2:
            return {"action": "end", "rationale": "Detected repeated auto-reply."}
        return {
            "action": "send",
            "body": (
                "Thanks. If you can connect me with the owner or manager, I can share "
                "the quick update. Reply YES or STOP."
            ),
            "cta": "yes_no",
            "rationale": "Auto-reply detected; trying one last handoff.",
        }

    recall_response = _reply_recall_due(trigger, normalized)
    if recall_response:
        return recall_response

    if _mentions_checklist(normalized):
        return {
            "action": "send",
            "body": "Sharing the checklist now. Want a 2-line summary for your team as well? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Merchant asked for compliance checklist.",
        }

    if _mentions_pricing(normalized):
        offer = _pick_offer(merchant, category)
        if offer:
            body_text = f"Should I use {offer} as the headline, or share a different price point?"
        else:
            body_text = "Share the top service + price, and I will draft the message." 
        return {
            "action": "send",
            "body": body_text,
            "cta": "open_ended",
            "rationale": "Merchant asked about pricing; collecting offer details.",
        }

    if _is_positive_intent(normalized):
        followup = _followup_for_trigger(trigger_kind, trigger, merchant, category, customer)
        return {
            "action": "send",
            "body": followup["body"],
            "cta": followup["cta"],
            "rationale": followup["rationale"],
        }

    if _looks_like_question(body.message, normalized):
        question_reply = _question_reply_for_trigger(trigger_kind, trigger, merchant, category)
        if question_reply:
            return question_reply
        return {
            "action": "send",
            "body": "Got it. Share the exact detail you want, and I will draft it right away.",
            "cta": "open_ended",
            "rationale": "Merchant asked a question; requesting the specific detail.",
        }

    fallback = _fallback_reply_for_trigger(trigger_kind, trigger, merchant, category, customer)
    if fallback:
        return fallback

    return {
        "action": "send",
        "body": "Noted. Should I proceed with a draft now? Reply YES or STOP.",
        "cta": "yes_no",
        "rationale": "Neutral reply; asking for a clear next step.",
    }


def compose(
    category: Dict[str, Any],
    merchant: Dict[str, Any],
    trigger: Dict[str, Any],
    customer: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kind = trigger.get("kind", "generic")
    send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    handlers = {
        "research_digest": _compose_research_digest,
        "regulation_change": _compose_regulation_change,
        "cde_opportunity": _compose_cde_opportunity,
        "perf_dip": _compose_perf_dip,
        "perf_spike": _compose_perf_spike,
        "seasonal_perf_dip": _compose_seasonal_perf_dip,
        "renewal_due": _compose_renewal_due,
        "festival_upcoming": _compose_festival,
        "ipl_match_today": _compose_ipl_match,
        "review_theme_emerged": _compose_review_theme,
        "milestone_reached": _compose_milestone,
        "curious_ask_due": _compose_curious_ask,
        "winback_eligible": _compose_winback,
        "dormant_with_vera": _compose_dormant,
        "gbp_unverified": _compose_gbp_unverified,
        "supply_alert": _compose_supply_alert,
        "category_seasonal": _compose_category_seasonal,
        "competitor_opened": _compose_competitor_opened,
        "active_planning_intent": _compose_active_planning,
        "recall_due": _compose_recall_due,
        "customer_lapsed_hard": _compose_customer_lapsed_hard,
        "trial_followup": _compose_trial_followup,
        "chronic_refill_due": _compose_chronic_refill,
        "wedding_package_followup": _compose_wedding_followup,
        "appointment_tomorrow": _compose_appointment_tomorrow,
    }

    body, cta, rationale = handlers.get(kind, _compose_generic)(
        category, merchant, trigger, customer
    )

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": trigger.get("suppression_key", ""),
        "rationale": rationale,
    }


def _compose_research_digest(category, merchant, trigger, customer):
    item = _find_digest_item(category, trigger)
    title = item.get("title") or "New research update"
    source = item.get("source")
    trial_n = item.get("trial_n")
    patient_segment = item.get("patient_segment")
    summary = _first_sentence(item.get("summary", ""))

    salutation = _merchant_salutation(merchant)
    anchor_bits = []
    if trial_n:
        anchor_bits.append(f"{trial_n}-patient trial")
    if patient_segment:
        anchor_bits.append(patient_segment.replace("_", " "))
    anchor = f" ({', '.join(anchor_bits)})" if anchor_bits else ""

    parts = [f"{salutation}, {title}{anchor}."]
    if summary:
        parts.append(summary)
    if source:
        parts.append(f"Source: {source}.")
    parts.append("Want me to pull the abstract and draft a patient message? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Research digest trigger with category digest item and merchant context."
    return body, "yes_no", rationale


def _compose_regulation_change(category, merchant, trigger, customer):
    item = _find_digest_item(category, trigger)
    title = item.get("title") or "Regulatory update"
    deadline = _safe_get(trigger, "payload", "deadline_iso")
    salutation = _merchant_salutation(merchant)

    parts = [f"{salutation}, {title}."]
    if deadline:
        parts.append(f"Effective from {deadline}.")
    actionable = _first_sentence(item.get("actionable", ""))
    if actionable:
        parts.append(actionable)
    parts.append("Want a 3-step checklist? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Compliance trigger with deadline and actionable guidance."
    return body, "yes_no", rationale


def _compose_cde_opportunity(category, merchant, trigger, customer):
    item = _find_digest_item(category, trigger)
    title = item.get("title") or "CDE opportunity"
    date = item.get("date") or _safe_get(trigger, "payload", "date")
    credits = _safe_get(trigger, "payload", "credits")
    salutation = _merchant_salutation(merchant)

    parts = [f"{salutation}, {title}."]
    if date:
        parts.append(f"Date: {date}.")
    if credits:
        parts.append(f"Credits: {credits}.")
    parts.append("Want the details link and summary? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "CDE opportunity with date and credits."
    return body, "yes_no", rationale


def _compose_perf_dip(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    metric = _safe_get(trigger, "payload", "metric") or "calls"
    delta_pct = _safe_get(trigger, "payload", "delta_pct")
    window = _safe_get(trigger, "payload", "window") or "7d"
    vs_baseline = _safe_get(trigger, "payload", "vs_baseline")
    delta_text = _format_pct(delta_pct)
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, your {metric} are {delta_text} over {window}."]
    if vs_baseline is not None:
        parts.append(f"Baseline: {vs_baseline}.")
    if offer:
        parts.append(f"Want me to draft a post for {offer} to lift {metric}? Reply YES or STOP.")
    else:
        parts.append("Want me to draft a quick post to lift this? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Performance dip trigger using metric delta and a concrete offer." 
    return body, "yes_no", rationale


def _compose_perf_spike(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    metric = _safe_get(trigger, "payload", "metric") or "calls"
    delta_pct = _safe_get(trigger, "payload", "delta_pct")
    window = _safe_get(trigger, "payload", "window") or "7d"
    delta_text = _format_pct(delta_pct, positive=True)
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, {metric} are up {delta_text} over {window}."]
    if offer:
        parts.append(f"Want me to amplify with a post for {offer}? Reply YES or STOP.")
    else:
        parts.append("Want me to amplify with a quick post? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Performance spike trigger to amplify momentum."
    return body, "yes_no", rationale


def _compose_seasonal_perf_dip(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    metric = _safe_get(trigger, "payload", "metric") or "views"
    delta_pct = _safe_get(trigger, "payload", "delta_pct")
    delta_text = _format_pct(delta_pct)
    offer = _pick_offer(merchant, category)

    parts = [
        f"{salutation}, {metric} are {delta_text} this week (seasonal dip).",
        "This is expected for the season.",
    ]
    if offer:
        parts.append(f"Want a quick post for {offer} to keep leads steady? Reply YES or STOP.")
    else:
        parts.append("Want a quick post to keep leads steady? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Seasonal dip trigger with reassurance and light action."
    return body, "yes_no", rationale


def _compose_renewal_due(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    days = _safe_get(trigger, "payload", "days_remaining")
    plan = _safe_get(trigger, "payload", "plan")
    amount = _safe_get(trigger, "payload", "renewal_amount")

    parts = [f"{salutation}, your {plan} renewal is due in {days} days."]
    if amount is not None:
        parts.append(f"Amount: INR {amount}.")
    parts.append("Want the renewal steps? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Renewal due trigger using plan and days remaining."
    return body, "yes_no", rationale


def _compose_festival(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    festival = _safe_get(trigger, "payload", "festival") or "festival"
    date = _safe_get(trigger, "payload", "date")
    days_until = _safe_get(trigger, "payload", "days_until")
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, {festival} is coming up."]
    if date:
        parts.append(f"Date: {date}.")
    if days_until is not None:
        parts.append(f"Days left: {days_until}.")
    if offer:
        parts.append(f"Want me to schedule a promo for {offer}? Reply YES or STOP.")
    else:
        parts.append("Want me to schedule a promo? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Festival trigger with date anchor and offer suggestion."
    return body, "yes_no", rationale


def _compose_ipl_match(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    match = _safe_get(trigger, "payload", "match") or "IPL match"
    time_iso = _safe_get(trigger, "payload", "match_time_iso")
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, IPL match today: {match}."]
    if time_iso:
        parts.append(f"Time: {time_iso}.")
    if offer:
        parts.append(f"Want me to push {offer} as a match-night delivery special? Reply YES or STOP.")
    else:
        parts.append("Want a match-night delivery special drafted? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "IPL match trigger with time and offer anchoring."
    return body, "yes_no", rationale


def _compose_review_theme(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    theme = _safe_get(trigger, "payload", "theme") or "service"
    occurrences = _safe_get(trigger, "payload", "occurrences_30d")
    quote = _safe_get(trigger, "payload", "common_quote")

    parts = [f"{salutation}, reviews mention '{theme}'."]
    if occurrences is not None:
        parts.append(f"Count in 30d: {occurrences}.")
    if quote:
        parts.append(f"Common quote: \"{quote}\".")
    parts.append("Want a response template and fix checklist? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Review theme trigger with specific occurrences and quote."
    return body, "yes_no", rationale


def _compose_milestone(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    metric = _safe_get(trigger, "payload", "metric") or "reviews"
    value_now = _safe_get(trigger, "payload", "value_now")
    milestone_value = _safe_get(trigger, "payload", "milestone_value")

    parts = [f"{salutation}, tracking your {metric} milestone."]
    if value_now is not None:
        parts[0] = f"{salutation}, you are at {value_now} {metric}."
    if milestone_value is not None and value_now is not None:
        remaining = milestone_value - value_now
        if remaining > 0:
            parts.append(f"Just {remaining} away from {milestone_value}.")
    parts.append("Want a quick review-ask message? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Milestone trigger with current value and target."
    return body, "yes_no", rationale


def _compose_curious_ask(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    body = (
        f"{salutation}, quick check: which service is most asked-for this week? "
        "I will turn it into a Google post and a 4-line WhatsApp reply."
    )
    rationale = "Curious ask trigger to drive engagement with low effort."
    return body, "open_ended", rationale


def _compose_winback(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    days_since_expiry = _safe_get(trigger, "payload", "days_since_expiry")
    lapsed_added = _safe_get(trigger, "payload", "lapsed_customers_added_since_expiry")
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, it has been {days_since_expiry} days since expiry."]
    if lapsed_added is not None:
        parts.append(f"{lapsed_added} new lapsed customers appeared since then.")
    if offer:
        parts.append(f"Want a winback message using {offer}? Reply YES or STOP.")
    else:
        parts.append("Want a winback message drafted? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Winback trigger with expiry and lapsed count."
    return body, "yes_no", rationale


def _compose_dormant(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    last_topic = _safe_get(trigger, "payload", "last_topic")
    body = (
        f"{salutation}, last time we discussed {last_topic}. "
        "Still want me to help with that, or should I focus on something else?"
    )
    rationale = "Dormant trigger prompting for the next best topic."
    return body, "open_ended", rationale


def _compose_gbp_unverified(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    uplift = _safe_get(trigger, "payload", "estimated_uplift_pct")
    path = _safe_get(trigger, "payload", "verification_path")

    parts = [f"{salutation}, your Google profile is still unverified."]
    if uplift is not None:
        parts.append(f"Estimated uplift: {int(uplift * 100)} percent.")
    if path:
        parts.append(f"Verification path: {path}.")
    parts.append("Want me to start verification? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "GBP unverified trigger with uplift estimate."
    return body, "yes_no", rationale


def _compose_supply_alert(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    molecule = _safe_get(trigger, "payload", "molecule")
    batches = _safe_get(trigger, "payload", "affected_batches") or []

    parts = [f"{salutation}, supply alert for {molecule}."]
    if batches:
        parts.append(f"Affected batches: {', '.join(batches)}.")
    parts.append("Want a customer list filtered for this molecule? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Supply alert trigger with molecule and batch list."
    return body, "yes_no", rationale


def _compose_category_seasonal(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    trends = _safe_get(trigger, "payload", "trends") or []
    trend_text = "; ".join(trends[:3])

    parts = [f"{salutation}, seasonal demand shift detected."]
    if trend_text:
        parts.append(f"Top moves: {trend_text}.")
    parts.append("Want a quick shelf or listing update plan? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Seasonal category trigger with trend list."
    return body, "yes_no", rationale


def _compose_competitor_opened(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    name = _safe_get(trigger, "payload", "competitor_name")
    distance = _safe_get(trigger, "payload", "distance_km")
    their_offer = _safe_get(trigger, "payload", "their_offer")
    offer = _pick_offer(merchant, category)

    parts = [f"{salutation}, new competitor nearby: {name} ({distance} km)."]
    if their_offer:
        parts.append(f"Their offer: {their_offer}.")
    if offer:
        parts.append(f"Want a counter-offer post for {offer}? Reply YES or STOP.")
    else:
        parts.append("Want a counter-offer post drafted? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Competitor trigger with distance and counter-offer suggestion."
    return body, "yes_no", rationale


def _compose_active_planning(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    intent_topic = _safe_get(trigger, "payload", "intent_topic") or "next campaign"

    body = (
        f"{salutation}, here is a starter outline for {intent_topic}. "
        "Share your target price and timing, and I will turn this into a draft post."
    )
    rationale = "Active planning intent trigger; moving to concrete drafting."
    return body, "open_ended", rationale


def _compose_recall_due(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your clinic"
    slots = _safe_get(trigger, "payload", "available_slots") or []
    service_due = _safe_get(trigger, "payload", "service_due") or "recall"
    service_due = service_due.replace("_", " ")
    offer = _pick_offer(merchant, category)

    parts = [f"Hi {name}, {merchant_name} here. Your {service_due} is due."]
    if slots:
        slot_labels = [s.get("label") for s in slots if s.get("label")]
        if len(slot_labels) >= 2:
            parts.append(f"Slots: 1) {slot_labels[0]}, 2) {slot_labels[1]}.")
            parts.append("Reply 1 or 2, or share a preferred time.")
        elif len(slot_labels) == 1:
            parts.append(f"Next available: {slot_labels[0]}. Reply YES to confirm.")
    if offer:
        parts.append(f"Offer: {offer}.")

    body = " ".join(parts)
    rationale = "Recall due trigger with slot options and offer reference."
    return body, "open_ended", rationale


def _compose_customer_lapsed_hard(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your gym"
    days = _safe_get(trigger, "payload", "days_since_last_visit")
    offer = _pick_offer(merchant, category)

    parts = [f"Hi {name}, {merchant_name} here."]
    if days is not None:
        parts.append(f"It has been {days} days since your last visit.")
    if offer:
        parts.append(f"We can book you with {offer}. Reply YES to restart.")
    else:
        parts.append("Want to restart with a short session this week? Reply YES or STOP.")

    body = " ".join(parts)
    rationale = "Customer lapsed hard trigger with time-since-last-visit."
    return body, "yes_no", rationale


def _compose_trial_followup(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your studio"
    options = _safe_get(trigger, "payload", "next_session_options") or []

    parts = [f"Hi {name}, {merchant_name} here. Want to continue after your trial?"]
    if options:
        label = options[0].get("label")
        if label:
            parts.append(f"Next slot: {label}. Reply YES to confirm or share a better time.")
    else:
        parts.append("Share a preferred time and we will book it.")

    body = " ".join(parts)
    rationale = "Trial follow-up trigger with next session option."
    return body, "open_ended", rationale


def _compose_chronic_refill(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your pharmacy"
    molecules = _safe_get(trigger, "payload", "molecule_list") or []
    stock_out = _safe_get(trigger, "payload", "stock_runs_out_iso")

    parts = [f"Hi {name}, {merchant_name} here."]
    if molecules:
        parts.append(f"Refill due for: {', '.join(molecules)}.")
    if stock_out:
        parts.append(f"Stock may run out by {stock_out}.")
    parts.append("Reply YES to schedule delivery or STOP to opt out.")

    body = " ".join(parts)
    rationale = "Chronic refill trigger with molecule list and stock-out date."
    return body, "yes_no", rationale


def _compose_wedding_followup(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your salon"
    days = _safe_get(trigger, "payload", "days_to_wedding")
    window = _safe_get(trigger, "payload", "next_step_window_open")

    parts = [f"Hi {name}, {merchant_name} here."]
    if days is not None:
        parts.append(f"Only {days} days to the wedding.")
    if window:
        parts.append(f"This is the right window for {window}.")
    parts.append("Want me to block your next slot? Reply YES or share a time.")

    body = " ".join(parts)
    rationale = "Wedding follow-up trigger with days-to-wedding."
    return body, "open_ended", rationale


def _compose_appointment_tomorrow(category, merchant, trigger, customer):
    name = _customer_name(customer)
    merchant_name = _safe_get(merchant, "identity", "name") or "your clinic"
    time_iso = _safe_get(trigger, "payload", "time_iso")

    parts = [f"Hi {name}, reminder from {merchant_name}."]
    if time_iso:
        parts.append(f"Your appointment is tomorrow at {time_iso}.")
    parts.append("Reply YES to confirm or STOP to reschedule.")

    body = " ".join(parts)
    rationale = "Appointment reminder trigger with time."
    return body, "yes_no", rationale


def _compose_generic(category, merchant, trigger, customer):
    salutation = _merchant_salutation(merchant)
    body = f"{salutation}, quick update from Vera. Want details? Reply YES or STOP."
    rationale = "Fallback trigger with safe CTA."
    return body, "yes_no", rationale


def _get_context(scope: str, context_id: str) -> Optional[Dict[str, Any]]:
    item = contexts.get((scope, context_id))
    return item.get("payload") if item else None


def _find_digest_item(category: Dict[str, Any], trigger: Dict[str, Any]) -> Dict[str, Any]:
    payload = trigger.get("payload", {})
    item_id = payload.get("top_item_id") or payload.get("digest_item_id")
    if not item_id:
        return payload.get("top_item", {}) or {}
    for item in category.get("digest", []):
        if item.get("id") == item_id:
            return item
    return {}


def _pick_offer(merchant: Dict[str, Any], category: Dict[str, Any]) -> Optional[str]:
    active = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    if active:
        return active[0].get("title")
    catalog = category.get("offer_catalog", [])
    if catalog:
        return catalog[0].get("title")
    return None


def _merchant_salutation(merchant: Dict[str, Any]) -> str:
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name") or "there"
    return f"Hi {name}"


def _customer_name(customer: Optional[Dict[str, Any]]) -> str:
    if not customer:
        return "there"
    return _safe_get(customer, "identity", "name") or "there"


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = text.split(".")
    return parts[0].strip() + "." if parts[0].strip() else ""


def _format_pct(delta: Optional[float], positive: bool = False) -> str:
    if delta is None:
        return "flat"
    pct = int(round(abs(delta) * 100))
    if positive:
        return f"{pct}%"
    return f"down {pct}%" if delta < 0 else f"up {pct}%"


def _template_params(merchant: Dict[str, Any], trigger: Dict[str, Any]) -> List[str]:
    name = _safe_get(merchant, "identity", "name") or "merchant"
    kind = trigger.get("kind", "update")
    return [str(name), str(kind)]


def _trigger_priority_score(trigger: Dict[str, Any]) -> int:
    urgency = int(trigger.get("urgency", 1)) * 10
    kind = trigger.get("kind", "generic")
    priority_map = {
        "active_planning_intent": 100,
        "recall_due": 90,
        "supply_alert": 90,
        "ipl_match_today": 85,
        "renewal_due": 80,
        "perf_dip": 70,
        "review_theme_emerged": 65,
        "perf_spike": 60,
        "competitor_opened": 60,
        "festival_upcoming": 40,
        "curious_ask_due": 30,
        "dormant_with_vera": 20,
    }
    return urgency + priority_map.get(kind, 10)


def _target_key(trigger: Dict[str, Any]) -> str:
    if trigger.get("scope") == "customer":
        return f"customer:{trigger.get('customer_id')}"
    return f"merchant:{trigger.get('merchant_id')}"


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _is_expired(trigger: Dict[str, Any], now: Optional[datetime]) -> bool:
    if not now:
        return False
    expires_at = trigger.get("expires_at")
    if not expires_at:
        return False
    exp = _parse_dt(expires_at)
    if not exp:
        return False
    if exp >= now:
        return False
    if trigger.get("kind") == "ipl_match_today":
        return now - exp > timedelta(hours=12)
    return True


def _normalize_text(text: str) -> str:
    return " ".join("".join(ch for ch in text.lower() if ch.isalnum() or ch.isspace()).split())


def _is_stop_message(normalized: str) -> bool:
    stop_phrases = [
        "stop",
        "unsubscribe",
        "not interested",
        "no thanks",
        "dont contact",
        "do not contact",
        "leave me",
        "remove me",
    ]
    return any(phrase in normalized for phrase in stop_phrases)


def _is_auto_reply(normalized: str, conv: Dict[str, Any]) -> bool:
    auto_phrases = [
        "thank you for contacting",
        "we have received",
        "automated response",
        "auto reply",
        "out of office",
        "this is an automated assistant",
        "we will get back",
    ]
    if any(phrase in normalized for phrase in auto_phrases):
        return True

    last_messages = [
        t.get("body", "") for t in conv.get("turns", []) if t.get("from") in ["merchant", "customer"]
    ]
    if len(last_messages) >= 2:
        return _normalize_text(last_messages[-1]) == _normalize_text(last_messages[-2])
    return False


def _is_repeat_in_conversation(conv: Dict[str, Any], normalized: str) -> bool:
    last_messages = [
        t.get("body", "") for t in conv.get("turns", []) if t.get("from") in ["merchant", "customer"]
    ]
    if len(last_messages) < 2:
        return False
    return _normalize_text(last_messages[-1]) == _normalize_text(last_messages[-2])


def _record_auto_reply(merchant_id: str, normalized: str) -> int:
    now = time.time()
    entry = auto_reply_tracker.get(merchant_id, {"text": "", "count": 0, "ts": 0.0})
    if entry["text"] != normalized or now - entry["ts"] > 3600:
        entry = {"text": normalized, "count": 1, "ts": now}
    else:
        entry["count"] += 1
        entry["ts"] = now
    auto_reply_tracker[merchant_id] = entry
    return entry["count"]


def _is_positive_intent(normalized: str) -> bool:
    positive_phrases = [
        "yes",
        "yeah",
        "yup",
        "ok",
        "okay",
        "sure",
        "go ahead",
        "please do",
        "send",
        "haan",
        "ha",
    ]
    return any(phrase in normalized for phrase in positive_phrases)


def _looks_like_question(raw: str, normalized: str) -> bool:
    question_words = ["what", "how", "price", "cost", "when", "why", "which"]
    return "?" in raw or any(word in normalized for word in question_words)


def _mentions_checklist(normalized: str) -> bool:
    return any(word in normalized for word in ["checklist", "audit", "compliance"])


def _mentions_pricing(normalized: str) -> bool:
    return any(word in normalized for word in ["price", "pricing", "rate", "cost", "offer"])


def _followup_for_trigger(
    trigger_kind: Optional[str],
    trigger: Optional[Dict[str, Any]],
    merchant: Dict[str, Any],
    category: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    if trigger_kind == "research_digest":
        return {
            "body": "Sharing the abstract now. Want a patient-facing WhatsApp draft too? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Merchant accepted research follow-up.",
        }
    if trigger_kind == "regulation_change":
        return {
            "body": "I can send a 3-step compliance checklist. Reply YES to receive it or STOP.",
            "cta": "yes_no",
            "rationale": "Merchant accepted compliance follow-up.",
        }
    if trigger_kind == "renewal_due":
        return {
            "body": "Great. Confirm your preferred payment method, and I will share renewal steps.",
            "cta": "open_ended",
            "rationale": "Merchant accepted renewal follow-up.",
        }
    if trigger_kind in ["perf_dip", "perf_spike", "seasonal_perf_dip"]:
        return {
            "body": "Which service should I highlight? Share the top one, and I will draft the post.",
            "cta": "open_ended",
            "rationale": "Merchant accepted performance follow-up.",
        }
    if trigger_kind == "review_theme_emerged":
        return {
            "body": "I will draft a response template. Want it in English or Hindi?",
            "cta": "open_ended",
            "rationale": "Merchant accepted review response follow-up.",
        }
    if trigger_kind == "competitor_opened":
        offer = _pick_offer(merchant, category)
        competitor = _safe_get(trigger or {}, "payload", "competitor_name") or "the new listing"
        return {
            "body": f"Got it. Should I counter {competitor} with {offer or 'a new offer'}?",
            "cta": "open_ended",
            "rationale": "Merchant accepted competitor follow-up.",
        }
    if trigger_kind == "ipl_match_today":
        offer = _pick_offer(merchant, category)
        match = _safe_get(trigger or {}, "payload", "match") or "today's match"
        return {
            "body": f"Ok. I will draft a {match} special using {offer or 'your top dish/offer'}. Delivery-only or dine-in focus?",
            "cta": "open_ended",
            "rationale": "Merchant accepted IPL match follow-up.",
        }
    if trigger_kind == "recall_due":
        slot_text = _slot_prompt(trigger)
        return {
            "body": slot_text,
            "cta": "open_ended",
            "rationale": "Merchant accepted recall follow-up; offering slots.",
        }
    return {
        "body": "Great. Tell me the top service and price to highlight, and I will draft it.",
        "cta": "open_ended",
        "rationale": "Generic positive intent follow-up.",
    }


def _question_reply_for_trigger(
    trigger_kind: Optional[str],
    trigger: Optional[Dict[str, Any]],
    merchant: Dict[str, Any],
    category: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    if trigger_kind == "regulation_change":
        deadline = _safe_get(trigger or {}, "payload", "deadline_iso")
        return {
            "action": "send",
            "body": f"Deadline is {deadline}. Want the 3-step checklist now? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Answered compliance question with deadline anchor.",
        }
    if trigger_kind == "review_theme_emerged":
        theme = _safe_get(trigger or {}, "payload", "theme") or "this issue"
        return {
            "action": "send",
            "body": f"I can draft a response for the {theme} reviews. English or Hindi?",
            "cta": "open_ended",
            "rationale": "Clarifying review response language.",
        }
    if trigger_kind == "ipl_match_today":
        offer = _pick_offer(merchant, category)
        return {
            "action": "send",
            "body": f"I can draft the match-night post. Use {offer or 'your top offer'} or a new price point?",
            "cta": "open_ended",
            "rationale": "Clarifying IPL offer.",
        }
    return None


def _reply_recall_due(trigger: Optional[Dict[str, Any]], normalized: str) -> Optional[Dict[str, Any]]:
    if not trigger or trigger.get("kind") != "recall_due":
        return None
    slots = _safe_get(trigger, "payload", "available_slots") or []
    slot_labels = [s.get("label") for s in slots if s.get("label")]
    if normalized in ["1", "one"] and len(slot_labels) >= 1:
        return {
            "action": "send",
            "body": f"Booked {slot_labels[0]}. Please confirm patient name and phone.",
            "cta": "open_ended",
            "rationale": "Slot selected for recall appointment.",
        }
    if normalized in ["2", "two"] and len(slot_labels) >= 2:
        return {
            "action": "send",
            "body": f"Booked {slot_labels[1]}. Please confirm patient name and phone.",
            "cta": "open_ended",
            "rationale": "Slot selected for recall appointment.",
        }
    if any(word in normalized for word in ["yes", "ok", "sure", "confirm"]):
        return {
            "action": "send",
            "body": _slot_prompt(trigger),
            "cta": "open_ended",
            "rationale": "Prompting for slot selection.",
        }
    return None


def _fallback_reply_for_trigger(
    trigger_kind: Optional[str],
    trigger: Optional[Dict[str, Any]],
    merchant: Dict[str, Any],
    category: Dict[str, Any],
    customer: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not trigger_kind:
        return None
    if trigger_kind == "ipl_match_today":
        offer = _pick_offer(merchant, category)
        return {
            "action": "send",
            "body": f"Should I draft a match-night post for {offer or 'your top offer'}? Delivery-only or dine-in focus?",
            "cta": "open_ended",
            "rationale": "Default IPL follow-up prompt.",
        }
    if trigger_kind == "regulation_change":
        deadline = _safe_get(trigger or {}, "payload", "deadline_iso")
        return {
            "action": "send",
            "body": f"Deadline is {deadline}. Want the 3-step checklist now? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default compliance follow-up prompt.",
        }
    if trigger_kind in ["perf_dip", "perf_spike", "seasonal_perf_dip"]:
        return {
            "action": "send",
            "body": "Which service and price should I highlight? Share one, and I will draft the post.",
            "cta": "open_ended",
            "rationale": "Default performance follow-up prompt.",
        }
    if trigger_kind == "review_theme_emerged":
        return {
            "action": "send",
            "body": "I can draft a response template. English or Hindi?",
            "cta": "open_ended",
            "rationale": "Default review follow-up prompt.",
        }
    if trigger_kind == "competitor_opened":
        offer = _pick_offer(merchant, category)
        return {
            "action": "send",
            "body": f"Want me to counter with {offer or 'a new offer'} or match their price?",
            "cta": "open_ended",
            "rationale": "Default competitor follow-up prompt.",
        }
    if trigger_kind == "recall_due":
        return {
            "action": "send",
            "body": _slot_prompt(trigger),
            "cta": "open_ended",
            "rationale": "Default recall follow-up prompt.",
        }
    if trigger_kind == "renewal_due":
        return {
            "action": "send",
            "body": "Want me to share renewal steps? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default renewal follow-up prompt.",
        }
    if trigger_kind == "supply_alert":
        return {
            "action": "send",
            "body": "Want the affected customer list now? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default supply alert follow-up prompt.",
        }
    if trigger_kind == "gbp_unverified":
        return {
            "action": "send",
            "body": "Should I start the verification flow now? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default GBP verification follow-up prompt.",
        }
    if trigger_kind == "research_digest":
        return {
            "action": "send",
            "body": "Want the abstract and a patient-facing WhatsApp draft? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default research digest follow-up prompt.",
        }
    if trigger_kind == "curious_ask_due":
        return {
            "action": "send",
            "body": "Which service is most asked-for this week? I will draft a post from it.",
            "cta": "open_ended",
            "rationale": "Default curious-ask follow-up prompt.",
        }
    if trigger_kind == "milestone_reached":
        return {
            "action": "send",
            "body": "Want a short review-ask message to hit the milestone? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default milestone follow-up prompt.",
        }
    if trigger_kind == "category_seasonal":
        return {
            "action": "send",
            "body": "Want a quick seasonal update plan for your listing? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default seasonal follow-up prompt.",
        }
    if trigger_kind == "active_planning_intent":
        return {
            "action": "send",
            "body": "Share target price and timing, and I will draft the outline.",
            "cta": "open_ended",
            "rationale": "Default planning follow-up prompt.",
        }
    if trigger_kind == "dormant_with_vera":
        return {
            "action": "send",
            "body": "Should I continue the last topic or switch to something else?",
            "cta": "open_ended",
            "rationale": "Default dormant follow-up prompt.",
        }
    if trigger_kind == "winback_eligible":
        return {
            "action": "send",
            "body": "Want a winback message drafted for your lapsed customers? Reply YES or STOP.",
            "cta": "yes_no",
            "rationale": "Default winback follow-up prompt.",
        }
    return None


def _slot_prompt(trigger: Optional[Dict[str, Any]]) -> str:
    slots = _safe_get(trigger or {}, "payload", "available_slots") or []
    slot_labels = [s.get("label") for s in slots if s.get("label")]
    if len(slot_labels) >= 2:
        return f"Reply 1 for {slot_labels[0]} or 2 for {slot_labels[1]}."
    if len(slot_labels) == 1:
        return f"Next available: {slot_labels[0]}. Reply YES to confirm or share a time."
    return "Share a preferred time and I will book it."


def _safe_get(data: Dict[str, Any], *keys: str) -> Optional[Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
