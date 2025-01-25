"use client";

import React, { useState, useEffect } from "react";
import { MetricsPanel } from "@/components/metrics-panel";
import { MetricsPanelToggle } from "@/components/metrics-panel-toggle";
import { ChatArea } from "@/components/ChatArea";
import { Information } from "@/components/Information";
import { SendSpace } from "@/components/SendSpace";
import { Message } from "@/lib/interfaces";
import { useMessages } from "@/hooks/useMessages";
import { useIdSession } from "@/hooks/useIdSession";
import { useSelectedOffices } from "@/hooks/useSelectedOffices";

import { ApplicationInsights } from "@microsoft/applicationinsights-web";
import { ReactPlugin } from "@microsoft/applicationinsights-react-js";
import { createBrowserHistory } from "history";

const reactPlugin = new ReactPlugin();
const browserHistory = createBrowserHistory();
const appInsights = new ApplicationInsights({
  config: {
    connectionString:
      "InstrumentationKey=981f8c2b-be12-40da-a719-0c3381c40ce1;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=006dc5ec-9bba-4c86-8e65-7e570a071152",

    // *** If you're adding the Click Analytics plug-in, delete the next line. ***
    extensions: [reactPlugin],
    // *** Add the Click Analytics plug-in. ***
    // extensions: [reactPlugin, clickPluginInstance],
    extensionConfig: {
      [reactPlugin.identifier]: { history: browserHistory },
      // *** Add the Click Analytics plug-in. ***
      // [clickPluginInstance.identifier]: clickPluginConfig
    },
  },
});
appInsights.loadAppInsights();
appInsights.trackPageView();

const DashboardComponent: React.FC = () => {
  const [isRightPanelOpen, setIsRightPanelOpen] = useState(true);

  const [messages, setMessages] = useState<Message[]>([]); // Lista de mensajes
  const [streamMessage, setStreamMessages] = useState<Message>({} as Message); // Mensaje stream, se eliminan al terminar y se agrega a la lista de mensajes
  const [loading, setLoading] = useState<boolean>(false);
  const [_, setIdSession] = useState<string>("");
  // Pass the selected offices to the right panel and to the chat system
  const [selectedOffices, setSelectedOffices] = useState<string[]>([]);
  const { add, read, update_feedback } = useMessages(setMessages);
  const { addOffice, readOffices, removeOffice } =
    useSelectedOffices(setSelectedOffices);
  const { addIdSession, readIdSession } = useIdSession(setIdSession);

  // Triger read from session storage
  useEffect(() => {
    setMessages(read()!);
    addIdSession(crypto.randomUUID()); // TODO: use a consistent session id
    setSelectedOffices(readOffices()); // Read the session statee
  }, []);

  // This is to trigger the effect when selecting a question
  const [selectedQuestion, setSelectedQuestion] = useState<string>("");

  return (
    <div className="container-layout p-4 transition-all duration-300 ease-in-out">
      <Information
        messages={messages}
        selectedOffices={selectedOffices}
        removeOffice={removeOffice}
      />
      <ChatArea
        messages={messages}
        stremMessage={streamMessage}
        loading={loading}
        update_feedback={update_feedback}
      />
      <SendSpace
        addMessage={add}
        setStreamMessages={setStreamMessages}
        setLoading={setLoading}
        loading={loading}
        readIdSession={readIdSession}
        selectedOffices={selectedOffices}
      />
      <MetricsPanelToggle
        isOpen={isRightPanelOpen}
        onToggle={() => setIsRightPanelOpen(!isRightPanelOpen)}
      />
      <MetricsPanel
        isOpen={isRightPanelOpen}
        setSelectedQuestion={setSelectedQuestion}
        selectedOffices={selectedOffices}
        setSelectedOffices={setSelectedOffices}
        addMessage={add}
        setStreamMessages={setStreamMessages}
        setLoading={setLoading}
        loading={loading}
        readIdSession={readIdSession}
      />
      <style jsx>{`
        .container-layout {
          display: grid;
          grid-template-columns: ${isRightPanelOpen ? "1fr" : "0fr"} 2.5em 4fr;
          grid-template-rows: 8em 1fr 4em;
          gap: 10px 10px;
          grid-auto-flow: row;
          height: 100vh;
          width: 100%;
          padding: 10px;

          grid-template-areas:
            "commands toggle-sidebar chat-info"
            "commands toggle-sidebar chat-area"
            "commands toggle-sidebar chat-input";
        }

        .office-selector {
          grid-area: office-selector;
        }

        .sys-info {
          grid-area: sys-info;
        }
      `}</style>
    </div>
  );
};

export default DashboardComponent;
