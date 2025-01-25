"use client";
import React from "react";
import Image from "next/image";
import { Badge } from "../ui/badge";
import logo from "@/lib/assets/Isotipo Blue White.svg";
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
    <div className="chat-info border border-zinc-800 rounded-lg p-4 bg-[#1c1c1c] shadow-lg">
      <div className="info-itself flex flex-col items-center space-y-2.5">
        <h3 className="font-semibold text-2xl text-amber-100 tracking-tight">
          Grokker™
        </h3>
        <p className="text-base font-medium text-zinc-400 tracking-wide">
          Multi-agentic workflow for data analysis
        </p>
        <div className="flex flex-wrap gap-1.5 mt-3 justify-center">
          <Badge className="bg-[#2c1810] text-amber-100 border-none shadow-sm">
            Toy dataset loaded
          </Badge>
          {selectedOffices &&
            selectedOffices
              .sort()
              .slice(0, LIMITE_OFICINAS)
              .map((office) => (
                <Badge
                  className="bg-[#1e2b32] text-zinc-300 hover:bg-[#243540] transition-colors"
                  key={office}
                  onClick={() => removeOffice(office)}
                >
                  {office}
                </Badge>
              ))}
          {selectedOffices.length > LIMITE_OFICINAS && (
            <Badge className="bg-[#3d2517] text-amber-100">
              {selectedOffices.length - LIMITE_OFICINAS} mas
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
};

export default Information;
