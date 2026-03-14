"""Unit tests for core.consumer.BaseConsumer."""

from unittest.mock import MagicMock, patch

import pytest
from pika.spec import Basic, BasicProperties

from core.consumer import BaseConsumer


class ConcreteConsumer(BaseConsumer):
    """Concrete consumer for testing; records on_message calls."""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.received: list[tuple] = []

    def on_message(self, channel, method, properties, body):  # noqa: ANN001
        self.received.append((channel, method, properties, body))


class TestBaseConsumerInit:
    """Tests for BaseConsumer.__init__."""

    def test_sets_required_attributes(self) -> None:
        consumer = ConcreteConsumer(
            "amqp://localhost/",
            "my_queue",
            "my_exchange",
            batch_size=5,
            routing_key="rk",
        )
        assert consumer.amqp_url == "amqp://localhost/"
        assert consumer.queue_name == "my_queue"
        assert consumer.exchange_name == "my_exchange"
        assert consumer.batch_size == 5
        assert consumer.routing_key == "rk"
        assert consumer.connection is None
        assert consumer.channel is None

    def test_routing_key_defaults_to_queue_name(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q1", "ex1")
        assert consumer.routing_key == "q1"

    def test_routing_key_none_uses_queue_name(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q1", "ex1", routing_key=None)
        assert consumer.routing_key == "q1"


class TestBaseConsumerSetup:
    """Tests for BaseConsumer._setup."""

    def test_setup_calls_channel_methods_with_correct_args(self) -> None:
        consumer = ConcreteConsumer(
            "amqp://x/", "q", "ex", batch_size=10, routing_key="rk"
        )
        channel = MagicMock()
        consumer.channel = channel

        consumer._setup()

        channel.exchange_declare.assert_called_once_with(
            exchange="ex", exchange_type="topic", durable=True
        )
        channel.queue_declare.assert_called_once_with(queue="q", durable=True)
        channel.queue_bind.assert_called_once_with(
            queue="q", exchange="ex", routing_key="rk"
        )
        channel.basic_qos.assert_called_once_with(prefetch_count=10)


class TestBaseConsumerOnMessageCallback:
    """Tests for BaseConsumer._on_message_callback."""

    def test_invokes_on_message_and_acks(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        method = Basic.Deliver(
            delivery_tag=123, redelivered=False, exchange="ex", routing_key="rk"
        )
        properties = BasicProperties()
        body = b"payload"

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 1
        assert consumer.received[0] == (channel, method, properties, body)
        channel.basic_ack.assert_called_once_with(delivery_tag=123)


class TestBaseConsumerStop:
    """Tests for BaseConsumer.stop."""

    def test_stop_when_channel_is_none(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        consumer.channel = None
        consumer.stop()  # no raise

    def test_stop_when_channel_open_calls_stop_consuming(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        channel.is_open = True
        consumer.channel = channel

        consumer.stop()

        channel.stop_consuming.assert_called_once()

    def test_stop_when_channel_closed_does_not_call_stop_consuming(
        self,
    ) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        channel.is_open = False
        consumer.channel = channel

        consumer.stop()

        channel.stop_consuming.assert_not_called()


class TestBaseConsumerRun:
    """Tests for BaseConsumer.run (with mocked pika)."""

    def test_run_setup_and_cleanup_on_exit(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_channel.is_open = True
        mock_connection.channel.return_value = mock_channel
        mock_channel.start_consuming.side_effect = KeyboardInterrupt

        with (
            patch(
                "core.consumer.pika.BlockingConnection", return_value=mock_connection
            ),
            patch("core.consumer.pika.URLParameters"),
        ):
            with pytest.raises(KeyboardInterrupt):
                consumer.run()

        mock_connection.channel.assert_called_once()
        mock_channel.exchange_declare.assert_called_once()
        mock_channel.queue_declare.assert_called_once()
        mock_channel.queue_bind.assert_called_once()
        mock_channel.basic_qos.assert_called_once()
        mock_channel.basic_consume.assert_called_once()
        mock_channel.close.assert_called_once()
        mock_connection.close.assert_called_once()
