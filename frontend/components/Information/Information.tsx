"use client";
import React from "react";
import { Badge } from "../ui/badge";
import "./Information.css";
import { Message } from "@/lib/interfaces";

export type InformationProps = {
  messages: Message[];
  selectedOffices: string[];
  removeOffice: (office: string) => void;
};

const Information: React.FC<InformationProps> = ({
  messages,
  selectedOffices,
  removeOffice,
}: InformationProps) => {
  const LIMITE_OFICINAS = 9000;

  return (
    <div className="flex flex-wrap gap-0.5 justify-start py-0.5 px-2 max-w-full min-h-[1.5rem] items-center h-6">
      <Badge className="bg-[#2c1810] text-amber-100 border-none shadow-sm h-3 text-[10px]">
        Grokker v0.0.1
      </Badge>
      {selectedOffices &&
        selectedOffices
          .sort()
          .slice(0, LIMITE_OFICINAS)
          .map((office) => (
            <Badge
              className="bg-[#8B4513] text-zinc-300 hover:bg-[#A0522D] transition-colors h-3 text-[10px]"
              key={office}
              onClick={() => removeOffice(office)}
            >
              {office}
            </Badge>
          ))}
      {selectedOffices.length > LIMITE_OFICINAS && (
        <Badge className="bg-[#2b1810] text-amber-100 h-3 text-[10px]">
          {selectedOffices.length - LIMITE_OFICINAS} mas
        </Badge>
      )}
    </div>
  );
};

export default Information;
