export function useSelectedOffices(setSelectedOffices: (x: string[]) => void) {
  const readOffices = (): string[] => {
    const data = sessionStorage.getItem("selectedOffices");
    return data ? JSON.parse(data) : [];
  };

  const addOffice = (office: string) => {
    const offices = readOffices();
    if (!offices.includes(office)) {
      const updated = [...offices, office];
      setSelectedOffices(updated);
      sessionStorage.setItem("selectedOffices", JSON.stringify(updated));
    }
  };

  const removeOffice = (office: string) => {
    const updated = readOffices().filter((item) => item !== office);
    setSelectedOffices(updated);
    sessionStorage.setItem("selectedOffices", JSON.stringify(updated));
  };

  return { addOffice, readOffices, removeOffice };
}
