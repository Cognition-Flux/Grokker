"use client";

import AppSettings from "@/lib/AppSettings.json";

import { ScrollArea } from "@/components/ui/scroll-area";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "./ui/input";
import { useSelectedOffices } from "@/hooks/useSelectedOffices";

interface Office {
  name: string;
  ref: string;
  region: string;
}

interface OfficeSelectorProps {
  onSelectionChange?: (selectedOffices: string[]) => void;
  defaultSelected: string[];
  selectedOffices: string[];
  setSelectedOffices?: (values: string[]) => void;
}

export function OfficeSelector({
  onSelectionChange,
  selectedOffices,
}: OfficeSelectorProps) {
  const [offices, setOffices] = useState<Office[]>([]);
  const [searchValue, setSearchValue] = useState("");

  useEffect(() => {
    fetch(AppSettings.BaseLlmUrl + "/offices", {
      headers: {
        // "Access-Control-Allow-Origin": "*",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        setOffices(data.offices);
      })
      .catch((error) => console.error("Error fetching offices:", error));
  }, []);

  // Add a reload button
  const reloadOffices = () => {
    fetch(AppSettings.BaseLlmUrl + "/offices", {
      headers: {
        // "Access-Control-Allow-Origin": "*",
      },
    })
      .then((response) => response.json())
      .then((data) => {
        setOffices(data.offices);
      })
      .catch((error) => console.error("Error fetching offices:", error));
  };

  const filteredOffices = offices.filter((office) =>
    searchValue
      ? office.name.toLowerCase().includes(searchValue.toLowerCase())
      : true
  );

  const { addOffice, readOffices, removeOffice } = useSelectedOffices(
    onSelectionChange!
  );

  // BEST CODE EVER
  const handleSelect = (officeName: string) => {
    const current = readOffices();

    current.includes(officeName)
      ? removeOffice(officeName)
      : addOffice(officeName);
  };

  useEffect(() => {
    if (onSelectionChange) {
      onSelectionChange(selectedOffices);
    }
  }, [selectedOffices, onSelectionChange]);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline">Oficinas</Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-96 overflow-y-auto max-h-[80%]">
        <Input
          type="string"
          placeholder="Oficina..."
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
        />

        <DropdownMenuSeparator />
        <ScrollArea className="h-[80vh] overflow-auto">
          {offices.length === 0 && (
            <div className="grid justify-items-center w-full">
              <Button variant="link" onClick={reloadOffices}>
                Recargar oficinas
              </Button>
            </div>
            // </DropdownMenuLabel>
          )}
          {offices.length !== 0 && filteredOffices.length === 0 && (
            <DropdownMenuLabel>Filtro sin oficinas</DropdownMenuLabel>
          )}
          {filteredOffices.map((office: Office) => (
            <DropdownMenuCheckboxItem
              checked={selectedOffices.includes(office.name)}
              onCheckedChange={() => handleSelect(office.name)}
              key={office.ref}
            >
              {office.name}
            </DropdownMenuCheckboxItem>
          ))}
        </ScrollArea>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
