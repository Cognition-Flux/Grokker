export function useIdSession(setIdSession: (x: string) => void) {

    const addIdSession = (idSession: string) => {
        setIdSession(idSession)
        sessionStorage.setItem('idSession', JSON.stringify(idSession));
    }

    const readIdSession = () => {
        let idSession = sessionStorage.getItem('idSession');
        setIdSession(idSession!)
        return JSON.parse(idSession!)
    }

    return { addIdSession, readIdSession }
}