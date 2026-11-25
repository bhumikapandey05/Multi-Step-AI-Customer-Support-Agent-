"""Fictional seed data. All people, emails (example.com), and PINs are made up.

Dates are relative to ``now`` so refund-window scenarios stay meaningful no
matter when the database is seeded. Each order exists to exercise a specific
policy path; see the ``scenario`` field.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from support_agent.crm.policies import RefundPolicy
from support_agent.crm.security import DEFAULT_PIN_ITERATIONS, hash_pin
from support_agent.db.models import (
    AccountStatus,
    Customer,
    CustomerTier,
    Order,
    OrderItem,
    OrderStatus,
    RefundReason,
    RefundRequest,
    RefundStatus,
    StaffMember,
    StaffRole,
    SupportPolicy,
    SupportTicket,
    TicketCategory,
    TicketPriority,
    TicketStatus,
)


@dataclass(frozen=True)
class DemoCustomer:
    customer_ref: str
    full_name: str
    email: str
    pin: str
    tier: CustomerTier = CustomerTier.STANDARD
    status: AccountStatus = AccountStatus.ACTIVE


DEMO_CUSTOMERS: tuple[DemoCustomer, ...] = (
    DemoCustomer("CUST-1001", "Alice Nguyen", "alice.nguyen@example.com", "246810"),
    DemoCustomer(
        "CUST-1002", "Brian Okafor", "brian.okafor@example.com", "135790", CustomerTier.GOLD
    ),
    DemoCustomer("CUST-1003", "Carmen Diaz", "carmen.diaz@example.com", "112233"),
    DemoCustomer("CUST-1004", "Daniel Kim", "daniel.kim@example.com", "445566"),
    DemoCustomer(
        "CUST-1005",
        "Emma Rossi",
        "emma.rossi@example.com",
        "778899",
        status=AccountStatus.SUSPENDED,
    ),
    DemoCustomer(
        "CUST-1006", "Farah Haddad", "farah.haddad@example.com", "990011", CustomerTier.GOLD
    ),
)

DEMO_PINS: dict[str, str] = {c.email: c.pin for c in DEMO_CUSTOMERS}


@dataclass(frozen=True)
class DemoStaff:
    staff_ref: str
    full_name: str
    email: str
    pin: str
    role: StaffRole


DEMO_STAFF: tuple[DemoStaff, ...] = (
    DemoStaff(
        "STF-001", "Sam Patel", "sam.agent@brightcart.example", "314159",
        StaffRole.SUPPORT_AGENT,
    ),
    DemoStaff(
        "STF-002", "Riley Chen", "riley.approver@brightcart.example", "271828",
        StaffRole.REFUND_APPROVER,
    ),
    DemoStaff(
        "STF-003", "Morgan Lee", "morgan.supervisor@brightcart.example", "161803",
        StaffRole.SUPERVISOR,
    ),
)  # fmt: skip

DEMO_STAFF_PINS: dict[str, str] = {s.email: s.pin for s in DEMO_STAFF}


@dataclass(frozen=True)
class _Item:
    sku: str
    name: str
    category: str
    unit_price_cents: int
    quantity: int = 1
    final_sale: bool = False


@dataclass(frozen=True)
class _Order:
    order_number: str
    email: str
    status: OrderStatus
    placed_days_ago: float
    shipped_days_ago: float | None
    delivered_days_ago: float | None
    items: tuple[_Item, ...]
    scenario: str


_ORDERS: tuple[_Order, ...] = (
    _Order(
        "ORD-10001",
        "alice.nguyen@example.com",
        OrderStatus.DELIVERED,
        9,
        7,
        5,
        (_Item("EL-1000", "Aurora Wireless Headphones", "electronics", 12_999),),
        "Eligible: delivered 5 days ago, inside the 30-day window.",
    ),
    _Order(
        "ORD-10002",
        "alice.nguyen@example.com",
        OrderStatus.DELIVERED,
        50,
        48,
        45,
        (
            _Item("AP-2201", "Trailblazer Rain Jacket", "apparel", 8_950),
            _Item("AP-2310", "Merino Wool Socks (3-pack)", "apparel", 1_800, quantity=2),
        ),
        "Not eligible: standard tier, delivered 45 days ago (window expired).",
    ),
    _Order(
        "ORD-10003",
        "alice.nguyen@example.com",
        OrderStatus.SHIPPED,
        4,
        2,
        None,
        (_Item("HM-3105", "Ceramic Pour-Over Coffee Set", "home", 4_500),),
        "Not eligible: shipped but not delivered yet.",
    ),
    _Order(
        "ORD-10004",
        "brian.okafor@example.com",
        OrderStatus.DELIVERED,
        52,
        49,
        45,
        (_Item("SP-4410", "Adjustable Dumbbell Set", "sports", 24_900),),
        "Eligible: gold tier gets a 60-day window, delivered 45 days ago.",
    ),
    _Order(
        "ORD-10005",
        "brian.okafor@example.com",
        OrderStatus.DELIVERED,
        15,
        13,
        10,
        (_Item("EL-5520", "Nimbus 14-inch Laptop", "electronics", 129_900),),
        "Eligible, but above the manual-review threshold.",
    ),
    _Order(
        "ORD-10006",
        "brian.okafor@example.com",
        OrderStatus.PROCESSING,
        1,
        None,
        None,
        (_Item("BT-6001", "Botanical Face Serum", "beauty", 3_200, quantity=2),),
        "Not eligible: not shipped yet (cancellation instead).",
    ),
    _Order(
        "ORD-10007",
        "carmen.diaz@example.com",
        OrderStatus.DELIVERED,
        8,
        6,
        3,
        (
            _Item("AP-7105", "Linen Summer Dress", "apparel", 6_400),
            _Item("AP-7990", "Clearance Sun Hat", "apparel", 1_500, final_sale=True),
        ),
        "Partially eligible: the final-sale hat is excluded.",
    ),
    _Order(
        "ORD-10008",
        "carmen.diaz@example.com",
        OrderStatus.DELIVERED,
        7,
        7,
        7,
        (
            _Item("DG-8001", "PhotoPro Editing Suite (1-year license)", "digital", 7_999),
            _Item("GC-0050", "$50 Gift Card", "gift_card", 5_000),
        ),
        "Not eligible: digital goods and gift cards only.",
    ),
    _Order(
        "ORD-10009",
        "carmen.diaz@example.com",
        OrderStatus.CANCELLED,
        6,
        None,
        None,
        (_Item("HM-9020", "Smart LED Desk Lamp", "home", 5_999),),
        "Not eligible: cancelled.",
    ),
    _Order(
        "ORD-10010",
        "daniel.kim@example.com",
        OrderStatus.DELIVERED,
        18,
        15,
        12,
        (_Item("EL-1150", "Pulse Fitness Tracker", "electronics", 9_999),),
        "Not eligible: a refund request is already pending review.",
    ),
    _Order(
        "ORD-10011",
        "daniel.kim@example.com",
        OrderStatus.DELIVERED,
        26,
        23,
        20,
        (_Item("HM-1120", "Bamboo Cutting Board Set", "home", 3_499),),
        "Not eligible: already fully refunded.",
    ),
    _Order(
        "ORD-10012",
        "emma.rossi@example.com",
        OrderStatus.DELIVERED,
        9,
        6,
        4,
        (_Item("SP-1201", "Yoga Mat Pro", "sports", 6_800),),
        "Eligible by policy, but the account is suspended (writes blocked).",
    ),
    _Order(
        "ORD-10013",
        "farah.haddad@example.com",
        OrderStatus.PENDING,
        0.1,
        None,
        None,
        (_Item("EL-1301", "Mechanical Keyboard", "electronics", 11_900),),
        "Not eligible: pending payment confirmation.",
    ),
    _Order(
        "ORD-10014",
        "farah.haddad@example.com",
        OrderStatus.DELIVERED,
        5,
        3,
        1,
        (_Item("HM-1401", "Cast Iron Dutch Oven", "home", 8_900),),
        "Eligible: delivered yesterday.",
    ),
)


def _policy_documents(policy: RefundPolicy, effective: date) -> list[SupportPolicy]:
    threshold = f"${policy.manual_review_threshold_cents / 100:,.2f}"
    docs = {
        "refund-policy": (
            "Refund Policy",
            "refunds",
            (
                f"Customers may request a refund within {policy.standard_window_days} days of "
                f"delivery. Gold members have {policy.gold_window_days} days. Refunds can only be "
                "requested for delivered orders; orders that have not shipped can be cancelled "
                "instead. Final-sale items, digital goods, and gift cards are not refundable. "
                "Only one refund request per order can be under review at a time. Every refund "
                "request is reviewed by the support team before any payment is issued, and "
                f"requests of {threshold} or more always require manual review."
            ),
        ),
        "final-sale-items": (
            "Final Sale Items",
            "refunds",
            (
                "Items marked final sale (for example clearance products) cannot be returned or "
                "refunded. Other items in the same order remain eligible under the refund policy."
            ),
        ),
        "digital-goods": (
            "Digital Goods and Gift Cards",
            "refunds",
            (
                "Software licenses, downloadable content, and gift cards are delivered instantly "
                "and are not refundable. Contact support if a code does not work."
            ),
        ),
        "shipping-and-delivery": (
            "Shipping and Delivery",
            "shipping",
            (
                "Orders ship within 2 business days. Standard delivery takes 3-7 business days "
                "after shipping. If an order has not arrived 10 days after shipping, open a "
                "shipping ticket so the team can investigate with the carrier."
            ),
        ),
        "order-cancellation": (
            "Order Cancellation",
            "orders",
            (
                "Orders that are pending or processing can be cancelled by the support team at no "
                "cost. Once an order has shipped it cannot be cancelled; request a refund after "
                "delivery instead."
            ),
        ),
        "support-tickets": (
            "Support Tickets and Response Times",
            "support",
            (
                "Target first-response times: urgent 4 hours, high 1 business day, normal 2 "
                "business days, low 5 business days. Only one open ticket per order and topic is "
                "kept; repeat requests are added to the existing ticket."
            ),
        ),
        "account-security": (
            "Account Security",
            "account",
            (
                "Support verifies your identity with your email and 6-digit support PIN before "
                "discussing orders. Support will never ask for your password or payment card "
                "number. After 5 incorrect PIN attempts the account is locked for 15 minutes."
            ),
        ),
    }
    return [
        SupportPolicy(
            slug=slug,
            title=title,
            category=category,
            body=body,
            version=policy.version,
            effective_date=effective,
        )
        for slug, (title, category, body) in docs.items()
    ]


@dataclass(frozen=True)
class SeedSummary:
    customers: int
    orders: int
    tickets: int
    refund_requests: int
    policies: int
    staff: int = 0


def seed_staff(
    session: Session, *, now: datetime, pin_iterations: int = DEFAULT_PIN_ITERATIONS
) -> int:
    """Insert demo staff that are missing. Safe to run on an existing database."""
    existing = set(session.scalars(select(StaffMember.email)))
    added = 0
    for demo in DEMO_STAFF:
        if demo.email in existing:
            continue
        session.add(
            StaffMember(
                staff_ref=demo.staff_ref,
                full_name=demo.full_name,
                email=demo.email,
                pin_hash=hash_pin(demo.pin, iterations=pin_iterations),
                role=demo.role,
                created_at=now - timedelta(days=200),
            )
        )
        added += 1
    session.flush()
    return added


def seed_database(
    session: Session,
    *,
    now: datetime,
    pin_iterations: int = DEFAULT_PIN_ITERATIONS,
    policy: RefundPolicy | None = None,
) -> SeedSummary:
    """Insert demo data into an empty database. Raises if customers already exist."""
    if session.scalar(select(func.count()).select_from(Customer)):
        raise RuntimeError("Database already contains customers; use a reset to re-seed")
    policy = policy or RefundPolicy()

    def ago(days: float | None) -> datetime | None:
        return None if days is None else now - timedelta(days=days)

    customers: dict[str, Customer] = {}
    for demo in DEMO_CUSTOMERS:
        customer = Customer(
            customer_ref=demo.customer_ref,
            full_name=demo.full_name,
            email=demo.email,
            pin_hash=hash_pin(demo.pin, iterations=pin_iterations),
            tier=demo.tier,
            status=demo.status,
            created_at=now - timedelta(days=400),
        )
        customers[demo.email] = customer
        session.add(customer)

    orders: dict[str, Order] = {}
    for spec in _ORDERS:
        order = Order(
            order_number=spec.order_number,
            customer=customers[spec.email],
            status=spec.status,
            placed_at=ago(spec.placed_days_ago),
            shipped_at=ago(spec.shipped_days_ago),
            delivered_at=ago(spec.delivered_days_ago),
            items=[
                OrderItem(
                    sku=i.sku,
                    product_name=i.name,
                    category=i.category,
                    quantity=i.quantity,
                    unit_price_cents=i.unit_price_cents,
                    is_final_sale=i.final_sale,
                )
                for i in spec.items
            ],
        )
        orders[spec.order_number] = order
        session.add(order)
    session.flush()

    refunds = [
        RefundRequest(
            refund_number="RFD-00000001",
            customer_id=orders["ORD-10010"].customer_id,
            order_id=orders["ORD-10010"].id,
            amount_cents=9_999,
            reason=RefundReason.DEFECTIVE,
            details="Tracker stopped charging after a week.",
            status=RefundStatus.PENDING_REVIEW,
            policy_version=policy.version,
            created_at=ago(2),
            updated_at=ago(2),
        ),
        RefundRequest(
            refund_number="RFD-00000002",
            customer_id=orders["ORD-10011"].customer_id,
            order_id=orders["ORD-10011"].id,
            amount_cents=3_499,
            reason=RefundReason.DAMAGED,
            details="Board arrived cracked.",
            status=RefundStatus.APPROVED,
            policy_version=policy.version,
            created_at=ago(15),
            updated_at=ago(13),
        ),
    ]
    tickets = [
        SupportTicket(
            ticket_number="TKT-00000001",
            customer_id=orders["ORD-10010"].customer_id,
            order_id=orders["ORD-10010"].id,
            category=TicketCategory.REFUND,
            priority=TicketPriority.NORMAL,
            status=TicketStatus.OPEN,
            subject="Fitness tracker will not charge",
            description="The tracker stopped charging after one week of use.",
            created_at=ago(2),
            updated_at=ago(2),
        ),
        SupportTicket(
            ticket_number="TKT-00000002",
            customer_id=orders["ORD-10002"].customer_id,
            order_id=orders["ORD-10002"].id,
            category=TicketCategory.SHIPPING,
            priority=TicketPriority.LOW,
            status=TicketStatus.RESOLVED,
            subject="Rain jacket arrived late",
            description="The jacket arrived three days after the estimated date.",
            created_at=ago(44),
            updated_at=ago(43),
        ),
    ]
    policies = _policy_documents(policy, effective=date(2026, 1, 1))
    session.add_all([*refunds, *tickets, *policies])
    session.flush()
    staff = seed_staff(session, now=now, pin_iterations=pin_iterations)

    return SeedSummary(
        customers=len(customers),
        orders=len(orders),
        tickets=len(tickets),
        refund_requests=len(refunds),
        policies=len(policies),
        staff=staff,
    )
