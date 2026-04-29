"""
scenarios.py — Evaluation scenarios with deterministic action verification.

Design principles:
  - Every task specifies positive checks (tool+args) AND negative checks (must_not_call) separately
  - Difficulty levels test progressively harder memory capabilities
  - Includes distractor facts, ambiguous inputs, and no-action cases
  - 10 tasks across 6 scenarios for meaningful statistical signal
"""

from dataclasses import dataclass, field
from enum import Enum


class Difficulty(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    CROSS_ENTITY = "cross_entity"
    NEGATIVE = "negative"           # distractors / should-not-act
    AMBIGUOUS = "ambiguous"          # underspecified instruction


VALID_TOOLS = frozenset({"send_email", "create_ticket", "schedule_meeting", "update_crm", "escalate", "no_action"})


@dataclass(frozen=True)
class ToolAssertion:
    """A single deterministic check on agent behavior."""
    tool_name: str
    required_args: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self):
        if self.tool_name not in VALID_TOOLS:
            raise ValueError(f"Unknown tool '{self.tool_name}'. Valid: {VALID_TOOLS}")


@dataclass(frozen=True)
class Task:
    instruction: str
    expect_calls: list[ToolAssertion]          # tools that MUST be called with these args
    expect_not_called: list[str] = field(default_factory=list)  # tools that MUST NOT be called
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_answer: str = ""
    required_facts: list[str] = field(default_factory=list)
    difficulty: Difficulty = Difficulty.SINGLE_HOP

    def __post_init__(self):
        invalid = [t for t in self.expect_not_called if t not in VALID_TOOLS]
        if invalid:
            raise ValueError(f"Unknown tool(s) in expect_not_called: {invalid}. Valid: {VALID_TOOLS}")


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    seed_facts: list[str]
    tasks: list[Task]


SCENARIOS: list[Scenario] = [

    # ── 1. Simple recall → single action ──
    Scenario(
        name="basic_recall_action",
        description="Recall a fact and take one correct action",
        seed_facts=[
            "Client: Acme Corp, contact: jane@acme.com, industry: healthcare, contract: $1.2M.",
            "Acme Corp's contract renews on 2025-06-15. Send renewal reminder 30 days before.",
            "Our account manager for Acme Corp is Sarah from the Sales team.",
            "DISTRACTOR: Office WiFi password is 'sunshine42'. IT contact: helpdesk@ourcompany.com.",
        ],
        tasks=[
            Task(
                instruction="It's May 16, 2025. Check if any contract renewals need attention.",
                expect_calls=[
                    ToolAssertion("send_email", {"to": "jane@acme.com"}, "Email Acme contact about renewal"),
                ],
                expect_not_called=["escalate", "create_ticket"],
                expected_answer_contains=["acme", "renewal", "june"],
                expected_answer="Acme Corp's contract renews June 15. Sent renewal reminder to jane@acme.com.",
                required_facts=["Acme Corp's contract renews on 2025-06-15. Send renewal reminder 30 days before."],
                difficulty=Difficulty.SINGLE_HOP,
            ),
            Task(
                instruction="Who manages the Acme Corp account?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information-only query — no action needed"),
                ],
                expect_not_called=["send_email", "create_ticket", "escalate", "schedule_meeting"],
                expected_answer_contains=["sarah", "sales"],
                expected_answer="Sarah from the Sales team manages the Acme Corp account.",
                required_facts=["Our account manager for Acme Corp is Sarah from the Sales team."],
                difficulty=Difficulty.SINGLE_HOP,
            ),
        ],
    ),

    # ── 2. Multi-hop → priority routing ──
    Scenario(
        name="multi_hop_routing",
        description="Connect facts across entities to route actions with correct priority",
        seed_facts=[
            "HealthFirst Inc: contact dr.chen@healthfirst.com, $1.2M contract, patient analytics platform.",
            "FinServe Ltd: contact mike@finserve.com, $800K contract, fraud detection system.",
            "HealthFirst reported critical bug in analytics dashboard on 2025-04-01. Unresolved.",
            "Bug severity policy: critical bugs get P1 tickets assigned to the on-call engineer.",
            "Current on-call engineer: Marcus (marcus@ourcompany.com).",
            "FinServe requested a feature for real-time alerts. Non-urgent.",
        ],
        tasks=[
            Task(
                instruction="Handle the outstanding HealthFirst issue.",
                expect_calls=[
                    ToolAssertion("create_ticket", {"priority": "P1", "assignee": "marcus"}, "P1 to on-call"),
                    ToolAssertion("send_email", {"to": "dr.chen@healthfirst.com"}, "Notify client"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["healthfirst", "critical", "marcus", "p1"],
                expected_answer="Created P1 ticket assigned to Marcus. Notified dr.chen@healthfirst.com.",
                required_facts=[
                    "HealthFirst reported critical bug in analytics dashboard on 2025-04-01. Unresolved.",
                    "Bug severity policy: critical bugs get P1 tickets assigned to the on-call engineer.",
                    "Current on-call engineer: Marcus (marcus@ourcompany.com).",
                ],
                difficulty=Difficulty.MULTI_HOP,
            ),
            Task(
                instruction="Handle the FinServe feature request.",
                expect_calls=[
                    ToolAssertion("create_ticket", {"priority": "P3"}, "Non-urgent → P3"),
                ],
                expect_not_called=["escalate", "schedule_meeting"],
                expected_answer_contains=["finserve", "feature"],
                expected_answer="Created P3 ticket for FinServe real-time alerts feature request.",
                required_facts=["FinServe requested a feature for real-time alerts. Non-urgent."],
                difficulty=Difficulty.MULTI_HOP,
            ),
        ],
    ),

    # ── 3. Temporal → must use latest data ──
    Scenario(
        name="temporal_updates",
        description="Agent must act on the newest fact, ignoring outdated ones",
        seed_facts=[
            "Acme Corp contact was john@acme.com until March 2025.",
            "UPDATE April 2025: Acme Corp new contact is lisa@acme.com (John left the company).",
            "Acme Corp has an open support ticket TK-442 from February 2025.",
            "UPDATE April 2025: TK-442 was resolved and closed.",
            "Policy: Escalate to VP if a client has more than 3 open tickets.",
            "Acme Corp currently has 1 open ticket (TK-501, low priority).",
        ],
        tasks=[
            Task(
                instruction="Send Acme Corp an update about their account status.",
                expect_calls=[
                    ToolAssertion("send_email", {"to": "lisa@acme.com"}, "Must use NEW contact lisa, not john"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["lisa", "acme"],
                expected_answer="Sent update to lisa@acme.com. 1 open ticket (TK-501), no escalation needed.",
                required_facts=[
                    "UPDATE April 2025: Acme Corp new contact is lisa@acme.com (John left the company).",
                    "Acme Corp currently has 1 open ticket (TK-501, low priority).",
                ],
                difficulty=Difficulty.TEMPORAL,
            ),
            Task(
                instruction="What is the current status of support ticket TK-442?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["send_email", "create_ticket", "escalate"],
                expected_answer_contains=["resolved", "closed"],
                expected_answer="TK-442 was resolved and closed in April 2025.",
                required_facts=["UPDATE April 2025: TK-442 was resolved and closed."],
                difficulty=Difficulty.TEMPORAL,
            ),
        ],
    ),

    # ── 4. Cross-entity synthesis → conditional branching ──
    Scenario(
        name="cross_entity_decisions",
        description="Synthesize across multiple clients to make different decisions per client",
        seed_facts=[
            "HealthFirst: $1.2M contract, healthcare, 3 open critical tickets, SLA breach risk.",
            "FinServe: $800K contract, fintech, 0 open tickets, satisfied.",
            "Nexus AI: $2M contract, AI/ML, 1 open ticket (P2), moderate satisfaction.",
            "Escalation policy: escalate to VP for clients with SLA breach risk.",
            "Quarterly review policy: schedule review meeting for all clients with contract > $1M.",
            "VP of Customer Success: diane@ourcompany.com.",
        ],
        tasks=[
            Task(
                instruction="Prepare for the quarterly review. Identify which clients need escalation and schedule review meetings for qualifying clients.",
                expect_calls=[
                    ToolAssertion("escalate", {"issue": "healthfirst"}, "HealthFirst SLA breach → escalate"),
                    ToolAssertion("schedule_meeting", {"topic": "review"}, "Reviews for >$1M clients"),
                ],
                expect_not_called=[],
                expected_answer_contains=["healthfirst", "escalat", "nexus", "review"],
                expected_answer="Escalated HealthFirst (SLA breach). Scheduled reviews for HealthFirst ($1.2M) and Nexus AI ($2M). FinServe ($800K) excluded.",
                required_facts=[
                    "HealthFirst: $1.2M contract, healthcare, 3 open critical tickets, SLA breach risk.",
                    "Nexus AI: $2M contract, AI/ML, 1 open ticket (P2), moderate satisfaction.",
                    "Escalation policy: escalate to VP for clients with SLA breach risk.",
                    "Quarterly review policy: schedule review meeting for all clients with contract > $1M.",
                ],
                difficulty=Difficulty.CROSS_ENTITY,
            ),
            Task(
                instruction="Which of our clients has the largest contract value?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["send_email", "create_ticket", "escalate", "schedule_meeting", "update_crm"],
                expected_answer_contains=["nexus", "2m"],
                expected_answer="Nexus AI has the largest contract at $2M.",
                required_facts=["Nexus AI: $2M contract, AI/ML, 1 open ticket (P2), moderate satisfaction."],
                difficulty=Difficulty.CROSS_ENTITY,
            ),
        ],
    ),

    # ── 5. Negative / distractor — should NOT act ──
    Scenario(
        name="negative_distractor",
        description="Agent must resist acting when conditions are not met",
        seed_facts=[
            "GlobalTech: $500K contract, 0 open tickets, all SLAs met, next review in 6 months.",
            "Policy: Only escalate if SLA is breached. Only send renewal reminders 30 days before renewal.",
            "GlobalTech contract renews on 2026-01-15.",
            "RUMOR: Someone mentioned GlobalTech might churn, but no formal risk flag exists.",
        ],
        tasks=[
            Task(
                instruction="It's May 2025. Check if GlobalTech needs any immediate attention.",
                expect_calls=[
                    ToolAssertion("no_action", {}, "No conditions met — must explicitly do nothing"),
                ],
                expect_not_called=["escalate", "send_email", "create_ticket", "schedule_meeting"],
                expected_answer_contains=["no"],
                expected_answer="No action needed. GlobalTech has 0 open tickets, SLAs met, renewal not until January 2026.",
                required_facts=[
                    "GlobalTech: $500K contract, 0 open tickets, all SLAs met, next review in 6 months.",
                    "GlobalTech contract renews on 2026-01-15.",
                ],
                difficulty=Difficulty.NEGATIVE,
            ),
        ],
    ),

    # ── 6. Ambiguous instruction + CRM update ──
    Scenario(
        name="ambiguous_crm_update",
        description="Agent must resolve ambiguity and perform a CRM update",
        seed_facts=[
            "Acme Corp: current tier is 'Silver', contract value $1.2M.",
            "Tier policy: clients with contracts >= $1M qualify for 'Gold' tier.",
            "Acme Corp's satisfaction score dropped from 8.5 to 6.2 last quarter.",
            "Policy: schedule check-in meeting if satisfaction drops below 7.0.",
            "Acme Corp account manager: Sarah (sarah@ourcompany.com).",
        ],
        tasks=[
            Task(
                instruction="Review Acme Corp's account and take any necessary actions based on current policies.",
                expect_calls=[
                    ToolAssertion("update_crm", {"company": "acme", "field": "tier", "value": "gold"}, "Upgrade tier to Gold"),
                    ToolAssertion("schedule_meeting", {"topic": "check-in"}, "Satisfaction < 7 → check-in"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["acme", "gold", "satisfaction"],
                expected_answer="Updated Acme Corp tier to Gold (contract $1.2M >= $1M threshold). Scheduled check-in meeting due to satisfaction drop to 6.2.",
                required_facts=[
                    "Acme Corp: current tier is 'Silver', contract value $1.2M.",
                    "Tier policy: clients with contracts >= $1M qualify for 'Gold' tier.",
                    "Acme Corp's satisfaction score dropped from 8.5 to 6.2 last quarter.",
                    "Policy: schedule check-in meeting if satisfaction drops below 7.0.",
                ],
                difficulty=Difficulty.AMBIGUOUS,
            ),
            Task(
                instruction="What is Acme Corp's current satisfaction score?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information-only query — no action needed"),
                ],
                expect_not_called=["update_crm", "send_email", "create_ticket"],
                expected_answer_contains=["6.2"],
                expected_answer="Acme Corp's satisfaction score is 6.2, down from 8.5 last quarter.",
                required_facts=["Acme Corp's satisfaction score dropped from 8.5 to 6.2 last quarter."],
                difficulty=Difficulty.SINGLE_HOP,
            ),
        ],
    ),
    # ── 7. Churn risk triage — multi-hop + cross-entity ──
    Scenario(
        name="churn_risk_triage",
        description="Detect churn signals by connecting satisfaction, ticket count, and policy across clients",
        seed_facts=[
            "RetailCo: $600K contract, contact anna@retailco.com, renewal in 45 days.",
            "RetailCo satisfaction score dropped to 5.1 this quarter (was 8.0 last quarter).",
            "RetailCo has 4 unresolved support tickets opened in the last 30 days.",
            "Churn risk policy: flag account for retention if satisfaction < 6.0 AND open tickets > 3.",
            "Retention policy: when an account is flagged, schedule a retention call with the account manager.",
            "RetailCo account manager: Tom (tom@ourcompany.com).",
            "DataSafe: $1.5M contract, contact cto@datasafe.io, satisfaction 8.5, 0 open tickets, all SLAs met.",
        ],
        tasks=[
            Task(
                instruction="Review all client accounts for churn risk and take necessary actions.",
                expect_calls=[
                    ToolAssertion("schedule_meeting", {"attendees": "tom"}, "Schedule retention call with RetailCo AM"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["retailco", "retention", "tom"],
                expected_answer="RetailCo flagged for churn risk (satisfaction 5.1, 4 open tickets). Scheduled retention call with Tom. DataSafe healthy — no action.",
                required_facts=[
                    "RetailCo satisfaction score dropped to 5.1 this quarter (was 8.0 last quarter).",
                    "RetailCo has 4 unresolved support tickets opened in the last 30 days.",
                    "Churn risk policy: flag account for retention if satisfaction < 6.0 AND open tickets > 3.",
                    "Retention policy: when an account is flagged, schedule a retention call with the account manager.",
                    "RetailCo account manager: Tom (tom@ourcompany.com).",
                ],
                difficulty=Difficulty.MULTI_HOP,
            ),
            Task(
                instruction="Is RetailCo at churn risk? Summarise the evidence.",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["send_email", "create_ticket", "escalate", "schedule_meeting"],
                expected_answer_contains=["retailco", "5.1", "4"],
                expected_answer="Yes. RetailCo satisfaction is 5.1 (below 6.0) and has 4 open tickets (above 3). Both churn conditions are met.",
                required_facts=[
                    "RetailCo satisfaction score dropped to 5.1 this quarter (was 8.0 last quarter).",
                    "RetailCo has 4 unresolved support tickets opened in the last 30 days.",
                    "Churn risk policy: flag account for retention if satisfaction < 6.0 AND open tickets > 3.",
                ],
                difficulty=Difficulty.MULTI_HOP,
            ),
        ],
    ),

    # ── 8. Account upgrade chain — multi-hop + ambiguous ──
    Scenario(
        name="account_upgrade_chain",
        description="Apply a chain of upgrade policies triggered by a client funding milestone",
        seed_facts=[
            "TechStart: contact bob@techstart.io, currently on Starter plan, annual contract $200K.",
            "TechStart announced Series B funding of $18M on 2025-03-15.",
            "Upgrade policy: move client to Growth plan when funding round exceeds $10M.",
            "Growth plan includes: dedicated success manager and monthly business reviews.",
            "Monthly business review policy: schedule within 30 days of plan upgrade.",
            "Success manager pool: Lisa (available), Raj (available), Wei (on leave).",
            "Assignment policy: when multiple success managers are available, assign alphabetically.",
            "DISTRACTOR: TechStart CEO posted about AI trends on LinkedIn. Not actionable.",
        ],
        tasks=[
            Task(
                instruction="TechStart just informed us of their Series B. Update their account accordingly.",
                expect_calls=[
                    ToolAssertion("update_crm", {"company": "techstart", "field": "plan", "value": "growth"}, "Upgrade to Growth plan"),
                    ToolAssertion("schedule_meeting", {"topic": "review"}, "Schedule monthly business review"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["techstart", "growth", "review"],
                expected_answer="Upgraded TechStart to Growth plan ($18M Series B exceeds $10M threshold). Scheduled monthly business review within 30 days. Assigned Lisa as success manager.",
                required_facts=[
                    "TechStart announced Series B funding of $18M on 2025-03-15.",
                    "Upgrade policy: move client to Growth plan when funding round exceeds $10M.",
                    "Monthly business review policy: schedule within 30 days of plan upgrade.",
                ],
                difficulty=Difficulty.MULTI_HOP,
            ),
            Task(
                instruction="Who should be assigned as TechStart's dedicated success manager, and why?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["send_email", "create_ticket", "escalate", "schedule_meeting", "update_crm"],
                expected_answer_contains=["lisa"],
                expected_answer="Lisa should be assigned. Lisa and Raj are both available (Wei is on leave). Alphabetically, Lisa comes before Raj.",
                required_facts=[
                    "Success manager pool: Lisa (available), Raj (available), Wei (on leave).",
                    "Assignment policy: when multiple success managers are available, assign alphabetically.",
                ],
                difficulty=Difficulty.MULTI_HOP,
            ),
            Task(
                instruction="What plan is TechStart currently on?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["update_crm", "send_email", "create_ticket", "escalate"],
                expected_answer_contains=["starter"],
                expected_answer="TechStart is currently on the Starter plan with a $200K annual contract.",
                required_facts=["TechStart: contact bob@techstart.io, currently on Starter plan, annual contract $200K."],
                difficulty=Difficulty.SINGLE_HOP,
            ),
        ],
    ),

    # ── 9. SLA incident response — temporal + multi-hop + negative ──
    Scenario(
        name="sla_incident_response",
        description="Derive SLA tier from incident timestamps, then apply correct post-incident policy",
        seed_facts=[
            "SkyBank: $2.5M contract, financial services, contact ops@skybank.com.",
            "2025-04-15 09:00 — SkyBank reports severe API latency, production impacted.",
            "2025-04-15 09:45 — Engineering identifies root cause: database connection pool exhausted.",
            "2025-04-15 10:15 — Hotfix deployed, SkyBank API fully restored. Incident closed.",
            "SLA policy: incidents resolved within 2 hours are P2. Incidents over 2 hours are P1 and require VP escalation.",
            "Post-incident policy: send a resolution summary to the client within 24 hours of resolution.",
            "VP of Engineering: priya@ourcompany.com.",
        ],
        tasks=[
            Task(
                instruction="The SkyBank incident on April 15 is now resolved. Handle any required follow-up.",
                expect_calls=[
                    ToolAssertion("send_email", {"to": "ops@skybank.com"}, "Send resolution summary per post-incident policy"),
                ],
                expect_not_called=["escalate"],
                expected_answer_contains=["skybank", "p2", "resolution"],
                expected_answer="Incident lasted 75 minutes (P2 — under 2-hour threshold). No VP escalation needed. Sent resolution summary to ops@skybank.com per post-incident policy.",
                required_facts=[
                    "2025-04-15 09:00 — SkyBank reports severe API latency, production impacted.",
                    "2025-04-15 10:15 — Hotfix deployed, SkyBank API fully restored. Incident closed.",
                    "SLA policy: incidents resolved within 2 hours are P2. Incidents over 2 hours are P1 and require VP escalation.",
                    "Post-incident policy: send a resolution summary to the client within 24 hours of resolution.",
                ],
                difficulty=Difficulty.TEMPORAL,
            ),
            Task(
                instruction="Should the SkyBank April 15 incident be escalated to VP?",
                expect_calls=[
                    ToolAssertion("no_action", {}, "Information query — no action needed"),
                ],
                expect_not_called=["escalate", "send_email", "create_ticket"],
                expected_answer_contains=["no", "p2"],
                expected_answer="No. The incident resolved in 75 minutes, below the 2-hour P1 threshold. It is classified as P2, which does not require VP escalation.",
                required_facts=[
                    "2025-04-15 09:00 — SkyBank reports severe API latency, production impacted.",
                    "2025-04-15 10:15 — Hotfix deployed, SkyBank API fully restored. Incident closed.",
                    "SLA policy: incidents resolved within 2 hours are P2. Incidents over 2 hours are P1 and require VP escalation.",
                ],
                difficulty=Difficulty.TEMPORAL,
            ),
        ],
    ),

]


def get_all_tasks() -> list[tuple[str, Task]]:
    return [(s.name, t) for s in SCENARIOS for t in s.tasks]


def get_task_count() -> int:
    return sum(len(s.tasks) for s in SCENARIOS)
