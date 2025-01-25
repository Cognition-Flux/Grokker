"use client";
import React from "react";
import "./LoadingAnimated.css";
import { useState, useEffect } from "react";

export type LoadingAnimatedProps = {
  setLoading: (x: boolean) => void;
};

const LoadingAnimated: React.FC<LoadingAnimatedProps> = ({ setLoading }) => {
  const [showButton, setShowButton] = useState(false);
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowButton(true); // Muestra el botón después de 5 segundos
    }, 10000); // Tiempo en milisegundos (5000 ms = 5 segundos)

    return () => clearTimeout(timer); // Limpia el temporizador al desmontar el componente
  }, []);
  const handleClick = () => {
    setLoading(false);
  };
  return (
    <div className="loader">
      <p className="text-amber-800">Trabajando</p>
      <div className="words">
        <span className="word text-cyan-600">llamando</span>
        <span className="word text-amber-600">llamando</span>
        <span className="word text-cyan-600">agentes</span>
        <span className="word text-amber-600">generando</span>
        <span className="word text-cyan-600">respuesta</span>
        <span className="word text-amber-600">generando</span>
      </div>
      {showButton && (
        <button className="cancelar" onClick={handleClick}>
          Un momento por favor, estamos trabajando en su requerimiento. Puede cancelar oprimiendo aquí.
        </button>
      )}
    </div>
  );
};

export default LoadingAnimated;
