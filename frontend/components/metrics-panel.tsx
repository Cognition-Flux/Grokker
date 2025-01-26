"use client";
import { OfficeSelector } from "@/components/multi-select";
import { PREGUNTAS_RAPIDAS } from "@/lib/mock_data";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList, // Do not delete
} from "./ui/command";
import { Card } from "./ui/card";
import { StreamData } from "@/lib/chat_api";
import { Message } from "@/lib/interfaces";
import { useState, useEffect } from "react";
import { useSelectedOffices } from "@/hooks/useSelectedOffices";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { AppSettings } from "@/lib/appSettings";

interface Office {
  name: string;
  ref: string;
  region: string;
}

interface MetricsPanelProps {
  isOpen: boolean;
  setSelectedQuestion: (question: string) => void;
  selectedOffices: string[];
  setSelectedOffices: (values: string[]) => void;
  addMessage: (icon: string, text: string) => void;
  setStreamMessages: (x: Message) => void;
  setLoading: (x: boolean) => void;
  readIdSession: () => string;
  loading: boolean;
}

export function MetricsPanel({
  isOpen,
  selectedOffices,
  setSelectedOffices,
  addMessage,
  setStreamMessages,
  setLoading,
  loading,
  readIdSession,
}: MetricsPanelProps) {
  const [offices, setOffices] = useState<Office[]>([]);
  const [searchOfficeValue, setSearchOfficeValue] = useState("");
  const [searchQuestionValue, setSearchQuestionValue] = useState("");

  // Fetch offices on component mount
  useEffect(() => {
    fetch(AppSettings.BaseLlmUrl + "/offices", {
      headers: {},
    })
      .then((response) => response.json())
      .then((data) => {
        setOffices(data.offices);
      })
      .catch((error) => console.error("Error fetching offices:", error));
  }, []);

  const handleQuestionSelect = (question: string) => {
    addMessage("human", question);
    StreamData(
      question,
      addMessage,
      setLoading,
      setStreamMessages,
      readIdSession,
      selectedOffices
    );
  };

  const customGroupBy = (array: any[]): Record<string, any[]> => {
    return array.reduce((result, item) => {
      const key = item.type;
      if (!result[key]) {
        result[key] = [];
      }
      result[key].push(item);
      return result;
    }, {} as Record<string, any[]>);
  };

  const { addOffice, readOffices, removeOffice } = useSelectedOffices(
    setSelectedOffices
  );

  const handleOfficeSelect = (officeName: string) => {
    const current = readOffices();
    current.includes(officeName) ? removeOffice(officeName) : addOffice(officeName);
  };

  const reloadOffices = () => {
    fetch(AppSettings.BaseLlmUrl + "/offices", {
      headers: {},
    })
      .then((response) => response.json())
      .then((data) => {
        setOffices(data.offices);
      })
      .catch((error) => console.error("Error fetching offices:", error));
  };

  const filteredOffices = offices.filter((office) =>
    searchOfficeValue
      ? office.name.toLowerCase().includes(searchOfficeValue.toLowerCase())
      : true
  );

  return (
    <div className="commands">
      {isOpen && (
        <div className="h-full overflow-auto flex flex-col gap-4 bg-[#1c1c1c] p-4 rounded-lg">
          <Command className="rounded-md bg-[#1c1c1c] border-none">
            <CommandInput 
              placeholder="Busca una pregunta rápida..." 
              value={searchQuestionValue}
              onValueChange={setSearchQuestionValue}
              className="text-zinc-300 border-none"
            />
            <CommandList className="text-zinc-400">
              <CommandEmpty>No se encontraron resultados.</CommandEmpty>
              {Object.entries(customGroupBy(PREGUNTAS_RAPIDAS)).map(
                ([type, preguntas]) => (
                  <CommandGroup heading={type} key={type} className="text-zinc-200">
                    {preguntas &&
                      preguntas.map((pregunta: any, index: any) => (
                        <CommandItem
                          key={index}
                          value={pregunta.question}
                          disabled={loading}
                          onSelect={() => handleQuestionSelect(pregunta.question)}
                          className="hover:bg-[#1e0f02] text-zinc-400"
                        >
                          {pregunta.question}
                        </CommandItem>
                      ))}
                  </CommandGroup>
                )
              )}
            </CommandList>
          </Command>

          <Command className="rounded-md bg-[#1c1c1c]">
            <CommandInput 
              placeholder="Buscar oficina..." 
              value={searchOfficeValue}
              onValueChange={setSearchOfficeValue}
              className="text-zinc-300"
            />
            <CommandList className="text-zinc-400">
              <CommandEmpty>
                <div className="grid justify-items-center w-full p-2">
                  <Button variant="ghost" onClick={reloadOffices} 
                    className="text-zinc-300 hover:text-zinc-200 hover:bg-[#1e0f02]">
                    Recargar oficinas
                  </Button>
                </div>
              </CommandEmpty>
              <CommandGroup heading="Oficinas" className="text-zinc-200">
                {filteredOffices.map((office) => (
                  <CommandItem
                    key={office.ref}
                    value={office.name}
                    onSelect={() => handleOfficeSelect(office.name)}
                    className="hover:bg-[#1e0f02] text-zinc-400"
                  >
                    <Checkbox 
                      checked={selectedOffices.includes(office.name)}
                      className="mr-2 border-zinc-600"
                    />
                    {office.name}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </div>
      )}
      <style jsx>{`
        .commands {
          grid-area: commands;
        }
      `}</style>
    </div>
  );
}
