"""Input validation for tool arguments."""

import pytest

from support_agent.crm.errors import InvalidInputError
from support_agent.crm.schemas import (
    CreateTicketInput,
    OrderLookupInput,
    RefundRequestInput,
    parse_input,
)

VALID_TICKET = {
    "category": "shipping",
    "subject": "Package is late",
    "description": "My package has not arrived yet.",
}


def test_order_number_is_normalized():
    assert parse_input(OrderLookupInput, {"order_number": "  ord-10001 "}).order_number == (
        "ORD-10001"
    )


@pytest.mark.parametrize("bad", ["10001", "ORD-1", "ORD-10001; DROP TABLE orders", ""])
def test_malformed_order_numbers_are_rejected(bad):
    with pytest.raises(InvalidInputError) as exc:
        parse_input(OrderLookupInput, {"order_number": bad})
    assert exc.value.details["errors"][0]["field"] == "order_number"


def test_unknown_fields_such_as_customer_id_are_rejected():
    with pytest.raises(InvalidInputError) as exc:
        parse_input(CreateTicketInput, {**VALID_TICKET, "customer_id": 2})
    assert {e["field"] for e in exc.value.details["errors"]} == {"customer_id"}


def test_refund_cannot_set_status_directly():
    with pytest.raises(InvalidInputError):
        parse_input(
            RefundRequestInput,
            {"order_number": "ORD-10001", "reason": "damaged", "status": "approved"},
        )


def test_text_fields_strip_control_characters_and_whitespace():
    ticket = parse_input(
        CreateTicketInput,
        {
            **VALID_TICKET,
            "subject": "  Package\x00 is\t\tlate  ",
            "description": "\x07 Line one\nLine two  ",
        },
    )
    assert ticket.subject == "Package is late"
    assert ticket.description == "Line one\nLine two"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("subject", "Hi"),
        ("subject", "x" * 121),
        ("description", "short"),
        ("category", "free_money"),
        ("priority", "critical"),
    ],
)
def test_ticket_field_constraints(field, value):
    with pytest.raises(InvalidInputError):
        parse_input(CreateTicketInput, {**VALID_TICKET, field: value})


@pytest.mark.parametrize("amount", [0, -100, 12.5, 10_000_001])
def test_refund_amount_must_be_positive_whole_cents(amount):
    with pytest.raises(InvalidInputError):
        parse_input(
            RefundRequestInput,
            {"order_number": "ORD-10001", "reason": "damaged", "amount_cents": amount},
        )


@pytest.mark.parametrize("key", ["short", "has spaces in it", "x" * 65])
def test_idempotency_key_format(key):
    with pytest.raises(InvalidInputError):
        parse_input(CreateTicketInput, {**VALID_TICKET, "idempotency_key": key})


def test_model_instances_pass_through():
    ticket = CreateTicketInput.model_validate(VALID_TICKET)
    assert parse_input(CreateTicketInput, ticket) is ticket
