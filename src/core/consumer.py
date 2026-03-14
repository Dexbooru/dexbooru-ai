import threading
from abc import ABC, abstractmethod

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.channel import Channel
from pika.spec import Basic, BasicProperties


def is_amqp_healthy(amqp_url: str) -> bool:
    try:
        conn = pika.BlockingConnection(pika.URLParameters(amqp_url))
        conn.close()
        return True
    except Exception:
        return False


class BaseConsumer(ABC, threading.Thread):
    """Abstract base consumer. Owns its own connection and channel for isolation."""

    def __init__(
        self,
        amqp_url: str,
        queue_name: str,
        exchange_name: str,
        *,
        batch_size: int = 1,
        routing_key: str | None = None,
    ):
        super().__init__()
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.batch_size = batch_size
        self.routing_key = routing_key or queue_name
        self.connection: BlockingConnection | None = None
        self.channel: BlockingChannel | None = None

    def _setup(self) -> None:
        assert self.channel is not None
        self.channel.exchange_declare(
            exchange=self.exchange_name, exchange_type="topic", durable=True
        )
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.queue_bind(
            queue=self.queue_name,
            exchange=self.exchange_name,
            routing_key=self.routing_key,
        )
        self.channel.basic_qos(prefetch_count=self.batch_size)

    def _on_message_callback(
        self,
        ch: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        self.on_message(ch, method, properties, body)
        ch.basic_ack(delivery_tag=method.delivery_tag)

    @abstractmethod
    def on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        """Handle a single message. Called by pika; child must implement."""
        ...

    def stop(self) -> None:
        if self.channel is not None and self.channel.is_open:
            self.channel.stop_consuming()

    def run(self) -> None:
        self.connection = pika.BlockingConnection(pika.URLParameters(self.amqp_url))
        self.channel = self.connection.channel()
        try:
            self._setup()
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._on_message_callback,
            )
            self.channel.start_consuming()
        finally:
            if self.channel is not None and self.channel.is_open:
                self.channel.close()
            if self.connection is not None and self.connection.is_open:
                self.connection.close()
