"""Unit tests for core.consumer.BaseConsumer."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from pika.spec import Basic, BasicProperties
from pydantic import BaseModel

from core.consumer import BaseConsumer


class ConcreteConsumer(BaseConsumer):
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.received: list[tuple] = []

    def on_message(self, channel, method, properties, body):  # noqa: ANN001
        self.received.append((channel, method, properties, body))


class AsyncConcreteConsumer(BaseConsumer):
    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.received: list[tuple] = []

    async def on_message(self, channel, method, properties, body):  # noqa: ANN001
        self.received.append((channel, method, properties, body))


class _TestMessageModel(BaseModel):
    value: int


class TestBaseConsumerInit:
    def test_sets_required_attributes(self) -> None:
        consumer = ConcreteConsumer(
            "amqp://localhost/",
            "my_queue",
            "my_exchange",
            routing_key="rk",
        )
        assert consumer.amqp_url == "amqp://localhost/"
        assert consumer.queue_name == "my_queue"
        assert consumer.exchange_name == "my_exchange"
        assert consumer.message_model is None
        assert consumer.routing_key == "rk"
        assert consumer.connection is None
        assert consumer.channel is None

    def test_sets_message_model_when_provided(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", message_model=_TestMessageModel)
        assert consumer.message_model is _TestMessageModel

    def test_routing_key_defaults_to_queue_name(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q1", "ex1")
        assert consumer.routing_key == "q1"

    def test_routing_key_none_uses_queue_name(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q1", "ex1", routing_key=None)
        assert consumer.routing_key == "q1"

    def test_injected_dependencies_become_attributes(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", my_service="injected", other=42)
        assert consumer.my_service == "injected"
        assert consumer.other == 42

    def test_reserved_attrs_not_overwritten_by_dependencies(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", connection=MagicMock(), channel=MagicMock())
        assert consumer.connection is None
        assert consumer.channel is None


class TestBaseConsumerSetup:
    def test_setup_calls_channel_methods_with_correct_args(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", routing_key="rk")
        channel = MagicMock()
        consumer.channel = channel

        consumer._setup()

        channel.exchange_declare.assert_called_once_with(exchange="ex", exchange_type="topic", durable=True)
        channel.queue_declare.assert_called_once_with(queue="q", durable=True)
        channel.queue_bind.assert_called_once_with(queue="q", exchange="ex", routing_key="rk")
        channel.basic_qos.assert_called_once_with(prefetch_count=1)


class TestBaseConsumerOnMessageCallback:
    def test_invokes_on_message_with_parsed_json_and_acks(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=123, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b'{"a": 1}'

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 1
        assert consumer.received[0][:3] == (channel, method, properties)
        assert consumer.received[0][3] == {"a": 1}
        channel.basic_ack.assert_called_once_with(delivery_tag=123)

    def test_invokes_on_message_with_validated_model_when_message_model_set(
        self,
    ) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", message_model=_TestMessageModel)
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=456, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b'{"value": 42}'

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 1
        payload = consumer.received[0][3]
        assert isinstance(payload, _TestMessageModel)
        assert payload.value == 42
        channel.basic_ack.assert_called_once_with(delivery_tag=456)

    def test_nacks_when_on_message_raises(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()

        def on_message_raise(*args, **kwargs):  # noqa: ANN002, ANN003
            raise ValueError("handler error")

        consumer.on_message = on_message_raise
        method = Basic.Deliver(delivery_tag=789, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b'{"x": 1}'

        consumer._on_message_callback(channel, method, properties, body)

        channel.basic_ack.assert_not_called()
        channel.basic_nack.assert_called_once_with(delivery_tag=789, requeue=False)

    def test_nacks_when_json_invalid(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=1, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b"not json"

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 0
        channel.basic_ack.assert_not_called()
        channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_nacks_when_validation_fails_with_message_model(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex", message_model=_TestMessageModel)
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=2, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b'{"value": "not an int"}'

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 0
        channel.basic_ack.assert_not_called()
        channel.basic_nack.assert_called_once_with(delivery_tag=2, requeue=False)

    def test_async_on_message_runs_and_acks_when_no_event_loop(self) -> None:
        consumer = AsyncConcreteConsumer("amqp://x/", "q", "ex")
        channel = MagicMock()
        method = Basic.Deliver(delivery_tag=100, redelivered=False, exchange="ex", routing_key="rk")
        properties = BasicProperties()
        body = b'{"a": 1}'

        consumer._on_message_callback(channel, method, properties, body)

        assert len(consumer.received) == 1
        assert consumer.received[0][3] == {"a": 1}
        channel.basic_ack.assert_called_once_with(delivery_tag=100)

    def test_async_on_message_runs_on_provided_event_loop_and_acks(
        self,
    ) -> None:
        import threading as th

        loop = asyncio.new_event_loop()

        def run_loop() -> None:
            loop.run_forever()

        thread = th.Thread(target=run_loop)
        thread.start()
        try:
            consumer = AsyncConcreteConsumer("amqp://x/", "q", "ex", event_loop=loop)
            channel = MagicMock()
            method = Basic.Deliver(
                delivery_tag=101,
                redelivered=False,
                exchange="ex",
                routing_key="rk",
            )
            properties = BasicProperties()
            body = b'{"a": 2}'

            consumer._on_message_callback(channel, method, properties, body)

            assert len(consumer.received) == 1
            assert consumer.received[0][3] == {"a": 2}
            channel.basic_ack.assert_called_once_with(delivery_tag=101)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_async_on_message_raises_nacks(self) -> None:
        consumer = AsyncConcreteConsumer("amqp://x/", "q", "ex")

        async def on_message_raise(*args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("async handler error")

        consumer.on_message = on_message_raise
        channel = MagicMock()
        method = Basic.Deliver(
            delivery_tag=102,
            redelivered=False,
            exchange="ex",
            routing_key="rk",
        )
        properties = BasicProperties()
        body = b'{"x": 1}'

        consumer._on_message_callback(channel, method, properties, body)

        channel.basic_ack.assert_not_called()
        channel.basic_nack.assert_called_once_with(delivery_tag=102, requeue=False)


class TestBaseConsumerStop:
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
    def test_run_setup_and_cleanup_on_exit(self) -> None:
        consumer = ConcreteConsumer("amqp://x/", "q", "ex")
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_channel.is_open = True
        mock_connection.channel.return_value = mock_channel
        mock_channel.start_consuming.side_effect = KeyboardInterrupt

        with (
            patch("core.consumer.pika.BlockingConnection", return_value=mock_connection),
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
