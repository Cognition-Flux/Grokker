# Import the `configure_azure_monitor()` function from the
# `azure.monitor.opentelemetry` package.
from azure.monitor.opentelemetry import configure_azure_monitor

# Import the tracing api from the `opentelemetry` package.
from opentelemetry import trace

# Configure OpenTelemetry to use Azure Monitor with the
# APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.
configure_azure_monitor(
    connection_string="InstrumentationKey=981f8c2b-be12-40da-a719-0c3381c40ce1;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=006dc5ec-9bba-4c86-8e65-7e570a071152",
    enable_live_metrics=True,
)

tracer = trace.get_tracer(__name__)


def run_empty() -> None:
    """Empty function, does nothing"""

    return
