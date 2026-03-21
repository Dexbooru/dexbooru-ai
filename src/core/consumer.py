import asyncio
import json
import threading
from abc import ABC, abstractmethod
from typing import Any

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.channel import Channel
from pika.spec import Basic, BasicProperties
from pydantic import BaseModel


def is_amqp_healthy(amqp_url: str) -> bool:
    try:
        conn = pika.BlockingConnection(pika.URLParameters(amqp_url))
        conn.close()
        return True
    except Exception:
        return False


_RESERVED_ATTRS = frozenset(
    {
        "amqp_url",
        "queue_name",
        "exchange_name",
        "message_model",
        "routing_key",
        "event_loop",
        "connection",
        "channel",
    }
)


class BaseConsumer(ABC, threading.Thread):
    def __init__(
        self,
        amqp_url: str,
        queue_name: str,
        exchange_name: str,
        *,
        message_model: type[BaseModel] | None = None,
        routing_key: str | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
        **dependencies: Any,
    ):
        super().__init__()
        self.amqp_url = amqp_url
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.message_model = message_model
        self.routing_key = routing_key or queue_name
        self.event_loop = event_loop
        self.connection: BlockingConnection | None = None
        self.channel: BlockingChannel | None = None
        for key, value in dependencies.items():
            if key not in _RESERVED_ATTRS:
                setattr(self, key, value)

    def _setup(self) -> None:
        assert self.channel is not None
        self.channel.exchange_declare(exchange=self.exchange_name, exchange_type="topic", durable=True)
        self.channel.queue_declare(queue=self.queue_name, durable=True)
        self.channel.queue_bind(
            queue=self.queue_name,
            exchange=self.exchange_name,
            routing_key=self.routing_key,
        )
        self.channel.basic_qos(prefetch_count=1)

    def _on_message_callback(
        self,
        ch: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        try:
            raw = body.decode("utf-8")
            payload: BaseModel | Any = json.loads(raw)
            if self.message_model is not None:
                payload = self.message_model.model_validate(payload)
            if asyncio.iscoroutinefunction(self.on_message):
                coro = self.on_message(ch, method, properties, payload)
                if self.event_loop is not None:
                    future = asyncio.run_coroutine_threadsafe(coro, self.event_loop)
                    future.result()
                else:
                    asyncio.run(coro)
            else:
                self.on_message(ch, method, properties, payload)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    @abstractmethod
    def on_message(
        self,
        channel: Channel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: BaseModel | Any,
    ) -> None: ...

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
