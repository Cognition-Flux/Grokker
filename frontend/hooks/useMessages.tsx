import { Message } from "@/lib/interfaces";

export function useMessages(setMessages: (x: Message[]) => void) {
  // TODO add cosa para run id
  const add = (
    icon: string,
    text: string,
    run_id: string | undefined = undefined,
    feedback: number = -1,
    feedback_text: string[] | undefined = undefined
  ) => {
    let messages: Message[] = read();
    let aux: Message = { icon, text, run_id, feedback, feedback_text };
    messages.push(aux);
    setMessages(messages); // Should be function
    sessionStorage.setItem("messages", JSON.stringify(messages));
  };

  const read = () => {
    let aux = sessionStorage.getItem("messages");
    if (aux === null) {
      return [];
    }
    return JSON.parse(aux!) as Message[];
  };

  const update_feedback = (
    run_id: string,
    feedback: number,
    feedback_text: string[] = []
  ) => {
    let messages: Message[] = read();
    let index = messages.findIndex((element) => element.run_id === run_id);
    if (index === -1) {
      console.log("Key for feedback not found");
      return;
    }

    // Update the feedback
    console.log("Updating feedback for run_id: ", run_id);
    messages[index].feedback = feedback;
    messages[index].feedback_text = feedback_text;

    // Save it
    setMessages(messages); // Should be function
    sessionStorage.setItem("messages", JSON.stringify(messages));

    return;
  };

  return { add, read, update_feedback };
}
