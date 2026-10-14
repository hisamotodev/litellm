"""Built-in audit sinks: the stdout log line and an OTel log record over OTLP. Each is one `AuditSink`."""

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState, set_span_in_context
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.control_plane.audit.ocsf import AuditRecord
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Undelivered

EVENT_NAME: Final = "litellm.control_plane.audit"
SCOPE: Final = InstrumentationScope(EVENT_NAME)
_VERDICT: Final = TypeAdapter(LogRecordExportResult)
LOG_PREFIX: Final = "control_plane.audit"
AUDIT_LOGGER: Final = verbose_proxy_logger.getChild(LOG_PREFIX)
AUDIT_LOGGER.setLevel(logging.INFO)


def ocsf_line(record: AuditRecord) -> str:
    return json.dumps(record.body, default=str, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LogSink:
    name: str = "log"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        for record in batch:
            AUDIT_LOGGER.info("%s %s", LOG_PREFIX, ocsf_line(record))
        return Delivered(len(batch))

    async def close(self) -> None:
        return None


def _severity(record: AuditRecord) -> tuple[str, SeverityNumber]:
    return ("INFO", SeverityNumber.INFO) if record.event.decision == "allowed" else ("WARN", SeverityNumber.WARN)


def _ids(record: AuditRecord) -> tuple[int, int]:
    trace_id: Final = int(record.event.trace_id, 16) if record.event.trace_id else 0
    span_id: Final = int(record.event.span_id, 16) if record.event.span_id else 0
    return trace_id, span_id


def _attributes(record: AuditRecord) -> Mapping[str, str | int]:
    return {
        "event.name": EVENT_NAME,
        "enduser.id": record.event.actor,
        "http.request.method": record.event.method,
        "url.path": record.event.path,
        "litellm.control_plane.decision": record.event.decision,
        "litellm.control_plane.dataplane": record.event.dataplane_id,
        "litellm.control_plane.audit.sequence": record.sequence,
        "litellm.control_plane.audit.hash": record.hash,
    }


def _log_data(record: AuditRecord, resource: Resource) -> ReadableLogRecord:
    trace_id, span_id = _ids(record)
    severity_text, severity_number = _severity(record)
    trace_flags: Final = TraceFlags(TraceFlags.SAMPLED if trace_id else TraceFlags.DEFAULT)
    return ReadableLogRecord(
        LogRecord(
            timestamp=int(record.event.at.timestamp() * 1_000_000_000),
            observed_timestamp=time.time_ns(),
            context=set_span_in_context(
                NonRecordingSpan(
                    SpanContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        is_remote=False,
                        trace_flags=trace_flags,
                        trace_state=TraceState(),
                    )
                )
            ),
            severity_text=severity_text,
            severity_number=severity_number,
            body=ocsf_line(record),
            attributes=dict(_attributes(record)),
        ),
        resource,
        SCOPE,
    )


@dataclass(frozen=True, slots=True)
class OtlpLogSink:
    """One OTel log record per audit event, body is the OCSF object, trace and span ids point at the request span.
    The batch goes straight to the exporter, off the event loop, so its verdict is the delivery result and a
    collector that refuses the batch makes the worker retry instead of the SDK dropping it later."""

    exporter: LogRecordExporter
    resource: Resource
    name: str = "otlp"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        data: Final = tuple(_log_data(record, self.resource) for record in batch)
        try:
            result: Final = await asyncio.to_thread(self.exporter.export, data)
        except Exception as error:  # noqa: BLE001  # an exporter that raises instead of returning FAILURE is still a failed delivery
            return Undelivered(f"otlp export raised {type(error).__name__}")
        match _VERDICT.validate_python(result):
            case LogRecordExportResult.SUCCESS:
                return Delivered(len(batch))
            case LogRecordExportResult.FAILURE:
                return Undelivered("otlp export failed")

    async def close(self) -> None:
        await asyncio.to_thread(self.exporter.shutdown)
